"""Learning mode (Stage 3): adapt a trained model to your own material (needs torch).

Input is audio with a beat grid aligned to it, in the layout `beatref eval` reads (`time
[beat-in-bar]` per line), such as the recordings the module's Record mode writes. The model's
top layers are fine-tuned on them, pulled towards the base weights (L2-SP) and optionally mixed
with replayed base training data so they don't forget everything else. The last part of each
recording is held back: the base model is scored on it first, and the adapted model is only
written once it beats that.

The tempo range and meter of the recordings are written to the model's `beatref.suggest` block, as
Config settings (the module offers to apply them).
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import Frontend
from .train import DEVICE, STRETCH, Data, Hyper, Net, calibrate, corpus, export, load_annotation


@dataclass
class Tune:
    layers: int = 3                # top layers to adapt (output first); 0 = all
    epochs: int = 8
    steps: int = 100
    lr: float = 1e-3
    l2sp: float = 1e-4             # pull towards the base weights
    replay: float = 0.0            # share of batches drawn from the replay sources
    holdout: float = 0.2           # tail of each recording held back for validation
    crop: int = 1000               # frames per training sequence
    seed: int = 0


def load_net(path):
    """Net and Hyper from an RTNeural JSON model written by `export` (standardisation stays folded)."""
    spec = json.loads(Path(path).read_text())
    hd = spec['beatref']['hyper']
    h = Hyper(**{k: tuple(v) if isinstance(v, list) else v for k, v in hd.items() if k in Hyper.__dataclass_fields__})
    net = Net(int(spec['in_shape'][-1]), h)
    L = spec['layers']
    put = lambda p, a: p.data.copy_(torch.as_tensor(np.asarray(a), dtype=torch.float32))
    put(net.inp.weight, np.asarray(L[0]['weights'][0]).T), put(net.inp.bias, L[0]['weights'][1])
    for c, layer in zip(net.convs, L[1:]):
        put(c.conv.weight, np.asarray(layer['weights'][0]).transpose(2, 1, 0)), put(c.conv.bias, layer['weights'][1])
    if net.gru:
        n, (W, U, b) = h.gru, L[-2]['weights']
        swap = lambda a: np.concatenate([a[n:2 * n], a[:n], a[2 * n:]])
        put(net.gru.weight_ih_l0, swap(np.asarray(W).T)), put(net.gru.weight_hh_l0, swap(np.asarray(U).T))
        put(net.gru.bias_ih_l0, swap(np.asarray(b[0]))), put(net.gru.bias_hh_l0, swap(np.asarray(b[1])))
    put(net.out.weight, np.asarray(L[-1]['weights'][0]).T), put(net.out.bias, L[-1]['weights'][1])
    return net, h, spec['beatref']


def suggest(files):
    """Preferred tempo range and meter that cover the recordings' grids."""
    bpm, meters = [], []
    for _, ann in files:
        beats, numbers = load_annotation(ann)
        bpm += list(60 / np.diff(beats))
        if numbers is not None and (numbers == 1).any():
            meters += list(numbers[np.r_[numbers[1:] == 1, False]].astype(int))
    lo, hi = np.percentile(bpm, [5, 95])
    lo, hi = round(float(lo) / 1.05, 1), round(float(hi) * 1.05, 1)
    s = dict(min_bpm=min(60.0, lo), max_bpm=max(200.0, hi), pref_min_bpm=lo, pref_max_bpm=hi)
    if meters:
        s['beats_per_bar'] = int(np.bincount(meters).argmax())
    return s


def split(data, holdout):
    """Cut every item at 1 - holdout of its length: (train items, validation items)."""
    tr, va = [], []
    for logbands, beat, down, has_down in data.items:
        k = int(len(logbands) * (1 - holdout))
        tr.append((logbands[:k], beat[:k], down[:k], has_down))
        va.append((logbands[k:], beat[k:], down[k:], has_down))
    return tr, va


def finetune(base, sources, out, t=Tune(), replay=(), cache='data/cache', jobs=None, threads=None):
    torch.set_num_threads(threads or os.cpu_count())
    net, h, meta = load_net(base)
    fe = Frontend.from_dict(h.frontend)
    h.crop = t.crop
    files = corpus(sources)
    if not files:
        raise SystemExit(f'no annotated audio found in {", ".join(sources)}')
    rng = np.random.default_rng(t.seed)
    torch.manual_seed(t.seed)

    data = Data(files, cache, fe, h.delay, jobs)
    data.items, held = split(data, t.holdout)
    va = Data([], cache, fe, h.delay)
    va.items = held
    mix = Data(corpus(replay), cache, fe, h.delay, jobs) if replay and t.replay > 0 else None
    minutes = sum(len(it[0]) for it in data.items[STRETCH.index(1.0)::len(STRETCH)]) / 6000
    print(f'{len(files)} recordings, {minutes:.1f} min to train on', flush=True)

    blocks = [net.out, *([net.gru] if net.gru else []), *reversed(net.convs), net.inp]
    blocks = blocks[:t.layers] if t.layers else blocks
    net.to(DEVICE)
    for p in net.parameters():
        p.requires_grad_(False)
    params = [p for b in blocks for p in b.parameters()]
    for p in params:
        p.requires_grad_(True)
    anchor = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, t.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t.epochs * t.steps)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    vrng = np.random.default_rng(1234)
    vbatches = [[a.to(DEVICE) for a in va.batch(vrng, h, fe)] for _ in range(8)]

    def validate():
        net.eval()
        with torch.no_grad():
            return sum(((loss_fn(net(x), y) * m).sum() / m.sum()).item() for x, y, m in vbatches) / len(vbatches)

    best = validate()
    print(f'base model: val {best:.4f}, adapting {sum(p.numel() for p in params)} parameters', flush=True)
    improved = False
    for epoch in range(t.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for _ in range(t.steps):
            x, y, m = (a.to(DEVICE) for a in (mix if mix and rng.random() < t.replay else data).batch(rng, h, fe))
            bce = (loss_fn(net(x), y) * m).sum() / m.sum()
            loss = bce + t.l2sp * sum(((p - a) ** 2).sum() for p, a in zip(params, anchor))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += bce.item()
        vl, mark = validate(), ''
        if vl < best:
            best, improved, mark = vl, True, ' *'
            export(net, h, fe, out, latency=meta['latency'], meta=dict(
                adapted=dict(base=Path(base).name, recordings=len(files), minutes=round(minutes, 2), tune=asdict(t)),
                suggest=suggest(files)))
        print(f'epoch {epoch + 1:3d}  train {total / t.steps:.4f}  val {vl:.4f}  {time.time() - t0:.0f} s{mark}', flush=True)
    if not improved:
        print('no epoch beat the base model on the held-back audio; nothing written', flush=True)
        return None
    calibrate(out, files, jobs)
    print(f'suggested settings: {json.loads(Path(out).read_text())["beatref"]["suggest"]}', flush=True)
    return best
