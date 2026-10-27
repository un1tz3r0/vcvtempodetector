"""Train the causal beat and downbeat network and export it as RTNeural JSON (needs torch).

Data: any directories `beatref eval` reads, annotated with `time [beat-in-bar]` per line. Features
are cached per file at several time-stretch factors (the frame hop is scaled, so tempo varies while
timbre doesn't); random gain is applied on the fly. Files are split into folds by a hash of their
name, so `--fold i/k` holds out the same files that `beatref eval --fold i/k` scores.

The network is sequential, so RTNeural's json_parser can build it at run time: a dense layer,
causal dilated convolutions (optionally a GRU), and a dense layer with sigmoid outputs (beat,
downbeat).
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config
from .evaluate import fold_of, load_audio, pairs, score_file
from .features import Features, Frontend, transform

STRETCH = (0.8, 0.9, 1.0, 1.12, 1.25)  # hop scale factors: < 1 slows the music down


@dataclass
class Hyper:
    width: int = 64
    kernel: int = 3
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128)  # receptive field 5.1 s
    gru: int = 0                   # optional GRU after the convolutions
    dropout: float = 0.1
    delay: int = 1                 # label frames after the first frame that contains the beat
    crop: int = 1500               # frames per training sequence
    batch: int = 16
    steps: int = 250               # batches per epoch
    epochs: int = 30
    lr: float = 2e-3
    gain_db: tuple = (-30.0, 6.0)
    seed: int = 0
    frontend: dict = field(default_factory=lambda: Frontend().to_dict())


def corpus(sources, fold=None, held_out=False):
    """(audio, annotation) pairs from 'root[:annotations]' sources, filtered by fold."""
    out = []
    for src in sources:
        root, _, ann = src.partition(':')
        out += list(pairs(root, ann or None))
    if fold:
        i, k = fold
        out = [p for p in out if (fold_of(p[0], k) == i) == held_out]
    return out


def load_annotation(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], (a[:, 1] if a.shape[1] > 1 else None)


def _cache(args):
    audio, ann, cache, fe = args
    out = Path(cache) / f'{audio.parent.name}-{audio.stem}.npz'
    if not out.exists():
        x, sr = load_audio(audio)
        f = Features(sr, fe)
        bands = {f'{s}': np.log10(f.bands(x, f.hop * s) + 1e-9).astype(np.float16) for s in STRETCH}
        np.savez(out, sr=sr, hop=f.hop, **bands)
    return out


class Data:
    """Cached features and frame targets for every file and stretch factor."""

    def __init__(self, files, cache, fe, delay, jobs=None):
        Path(cache).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(jobs) as pool:
            paths = list(pool.map(_cache, [(a, n, cache, fe) for a, n in files], chunksize=4))
        self.items = []
        for (audio, ann), path in zip(files, paths):
            z = np.load(path)
            beats, numbers = load_annotation(ann)
            for s in STRETCH:
                logbands = z[f'{s}']
                frames_per_s = float(z['sr']) / (float(z['hop']) * s)
                n = len(logbands)
                beat, down = np.zeros(n, np.float32), np.zeros(n, np.float32)
                k = np.floor(beats * frames_per_s).astype(int) + delay
                for target, idx in ((beat, k), (down, k[numbers == 1] if numbers is not None else k[:0])):
                    for off, v in ((-1, .5), (1, .5), (0, 1.)):  # widened: neighbours count half
                        j = idx + off
                        j = j[(j >= 0) & (j < n)]
                        target[j] = np.maximum(target[j], v)
                self.items.append((logbands, beat, down, numbers is not None))

    def batch(self, rng, h, fe):
        L = h.crop
        x = np.zeros((h.batch, L, fe.size), np.float32)
        y = np.zeros((h.batch, L, 2), np.float32)
        m = np.zeros((h.batch, L, 2), np.float32)
        for b in range(h.batch):
            logbands, beat, down, has_down = self.items[rng.integers(len(self.items))]
            n = len(logbands)
            s = rng.integers(max(n - L, 0) + 1)
            e = min(s + L, n)
            gain = 10 ** (rng.uniform(*h.gain_db) / 20)
            # the sequence starts cold, as the module does after a reset
            x[b, :e - s] = transform(10 ** logbands[s:e].astype(np.float32), fe, gain)
            y[b, :e - s, 0], y[b, :e - s, 1] = beat[s:e], down[s:e]
            m[b, :e - s, 0], m[b, :e - s, 1] = 1, float(has_down)
        return map(torch.from_numpy, (x, y, m))


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad, self.conv = (k - 1) * d, nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):  # (batch, time, channels)
        return self.conv(nn.functional.pad(x.transpose(1, 2), (self.pad, 0))).transpose(1, 2)


class Net(nn.Module):
    def __init__(self, inputs, h):
        super().__init__()
        self.register_buffer('mean', torch.zeros(inputs))
        self.register_buffer('std', torch.ones(inputs))
        self.inp = nn.Linear(inputs, h.width)
        self.convs = nn.ModuleList(CausalConv(h.width, h.width, h.kernel, d) for d in h.dilations)
        self.gru = nn.GRU(h.width, h.gru, batch_first=True) if h.gru else None
        self.out = nn.Linear(h.gru or h.width, 2)
        self.drop = nn.Dropout(h.dropout)

    def forward(self, x):
        x = self.drop(nn.functional.elu(self.inp((x - self.mean) / self.std)))
        for c in self.convs:
            x = self.drop(nn.functional.elu(c(x)))
        if self.gru:
            x = self.drop(self.gru(x)[0])
        return self.out(x)  # logits


def export(net, h, fe, path, latency, meta=None):
    """Write RTNeural JSON (Keras layout), with the input standardisation folded into the first layer."""
    t = lambda a: a.detach().double().numpy()
    W, b = t(net.inp.weight) / t(net.std), t(net.inp.bias)
    b = b - W @ t(net.mean)
    layers = [dict(type='dense', activation='elu', shape=[None, None, h.width], weights=[W.T.tolist(), b.tolist()])]
    for c in net.convs:
        w = t(c.conv.weight)  # [out][in][k] -> [k][in][out]
        layers.append(dict(type='conv1d', activation='elu', shape=[None, None, h.width],
                           kernel_size=[h.kernel], dilation=[c.conv.dilation[0]], groups=1,
                           weights=[w.transpose(2, 1, 0).tolist(), t(c.conv.bias).tolist()]))
    if net.gru:
        n = h.gru
        swap = lambda a: np.concatenate([a[n:2 * n], a[:n], a[2 * n:]])  # torch r, z, n -> Keras z, r, h
        layers.append(dict(type='gru', activation='', shape=[None, None, n], weights=[
            swap(t(net.gru.weight_ih_l0)).T.tolist(), swap(t(net.gru.weight_hh_l0)).T.tolist(),
            [swap(t(net.gru.bias_ih_l0)).tolist(), swap(t(net.gru.bias_hh_l0)).tolist()]]))
    layers.append(dict(type='dense', activation='sigmoid', shape=[None, None, 2],
                       weights=[t(net.out.weight).T.tolist(), t(net.out.bias).tolist()]))
    for layer in layers:  # float32 precision halves the file
        layer['weights'] = json.loads(json.dumps(layer['weights']), parse_float=lambda v: float(f'{float(v):.8g}'))
    spec = dict(in_shape=[None, None, fe.size], layers=layers,
                beatref=dict(frontend=fe.to_dict(), outputs=['beat', 'downbeat'], latency=latency,
                             hyper=asdict(h), **(meta or {})))
    Path(path).write_text(json.dumps(spec, separators=(',', ':')))


def train(sources, out, h=Hyper(), fold=None, cache='data/cache', jobs=None, threads=None):
    torch.set_num_threads(threads or os.cpu_count())
    fe = Frontend.from_dict(h.frontend)
    files = corpus(sources, fold)
    rng = np.random.default_rng(h.seed)
    torch.manual_seed(h.seed)
    val = set(rng.choice(len(files), max(1, len(files) // 10), replace=False))
    t0 = time.time()
    tr = Data([f for i, f in enumerate(files) if i not in val], cache, fe, h.delay, jobs)
    va = Data([f for i, f in enumerate(files) if i in val], cache, fe, h.delay, jobs)
    print(f'{len(files) - len(val)} training and {len(val)} validation files, features in {time.time() - t0:.0f} s', flush=True)

    net = Net(fe.size, h)
    sample = np.concatenate([transform(10 ** lb.astype(np.float32), fe) for lb, *_ in tr.items[::37]])
    net.mean.copy_(torch.from_numpy(sample.mean(0)))
    net.std.copy_(torch.from_numpy(sample.std(0) + 1e-3))
    # start from the base rates: from zero logits, training sits on a plateau for many epochs
    prior = np.array([np.mean(np.concatenate([it[k] for it in tr.items[::7]])) for k in (1, 2)])
    net.out.bias.data.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())
    opt = torch.optim.AdamW(net.parameters(), h.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, h.lr, total_steps=h.epochs * h.steps, pct_start=0.15)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    vrng = np.random.default_rng(1234)
    vbatches = [list(va.batch(vrng, h, fe)) for _ in range(max(4, len(va.items) // (h.batch * 2)))]
    best = np.inf
    print(f'{sum(p.numel() for p in net.parameters())} parameters', flush=True)

    for epoch in range(h.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for _ in range(h.steps):
            x, y, m = tr.batch(rng, h, fe)
            loss = (loss_fn(net(x), y) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        net.eval()
        with torch.no_grad():
            vl = sum(((loss_fn(net(x), y) * m).sum() / m.sum()).item() for x, y, m in vbatches) / len(vbatches)
        mark = ''
        if vl < best:
            best, mark = vl, ' *'
            export(net, h, fe, out, latency=(h.delay + 0.5) / 100, meta=dict(fold=fold and f'{fold[0]}/{fold[1]}'))
        print(f'epoch {epoch + 1:3d}  train {total / h.steps:.4f}  val {vl:.4f}  {time.time() - t0:.0f} s{mark}', flush=True)
    calibrate(out, [f for i, f in enumerate(files) if i in val], jobs)
    return best


def calibrate(path, files, jobs=None):
    """Set the model's latency so the tracked beats on `files` land on the annotations on average."""
    spec = json.loads(Path(path).read_text())
    with ProcessPoolExecutor(jobs) as pool:
        rows = list(pool.map(score_file, repeat(Config(model=str(path))), repeat('py'), *zip(*files)))
    offset = float(np.nanmean([r['offset_ms'] for r in rows])) / 1000
    spec['beatref']['latency'] = max(0.0, spec['beatref']['latency'] + offset)  # early beats: less compensation
    Path(path).write_text(json.dumps(spec, separators=(',', ':')))
    print(f'latency {spec["beatref"]["latency"] * 1000:.1f} ms (beats were {offset * 1000:+.1f} ms off)', flush=True)
