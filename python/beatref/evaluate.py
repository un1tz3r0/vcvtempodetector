"""Evaluation: standard beat metrics (mir_eval) plus the ones that matter for a live clock."""
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import repeat
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from .cpp import detect_cpp
from .detector import detect

AUDIO = {'.wav', '.flac', '.aiff', '.aif', '.au', '.ogg', '.mp3'}
WINDOW = 0.07  # match tolerance [s], as in mir_eval's F-measure


def load_audio(path):
    x, sr = sf.read(path, always_2d=True)
    return x.mean(1), sr


def load_beats(path):
    return np.loadtxt(path, ndmin=2)[:, 0]


def load_downbeats(path):
    a = np.loadtxt(path, ndmin=2)
    return a[a[:, 1] == 1, 0] if a.shape[1] > 1 else None


def load_meter(path):
    """Beats per bar from beat-in-bar numbers (the most common bar length), or None."""
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] < 2 or not (a[:, 1] == 1).any():
        return None
    return int(np.bincount(a[np.r_[a[1:, 1] == 1, False], 1].astype(int)).argmax()) or None


def pairs(root, annotations=None):
    """(audio, annotation) pairs: <stem>.beats or <stem>.txt beside the audio or in `annotations`."""
    root = Path(root)
    dirs = [Path(annotations)] if annotations else []
    for audio in sorted(p for p in root.rglob('*') if p.suffix.lower() in AUDIO):
        for d in dirs + [audio.parent]:
            ann = next((d / (audio.stem + s) for s in ('.beats', '.txt') if (d / (audio.stem + s)).exists()), None)
            if ann:
                yield audio, ann
                break


def fold_of(path, k):
    """Cross-validation fold of a file, from a hash of its name."""
    return zlib.crc32(Path(path).stem.encode()) % k


def parse_fold(spec):
    """'i/k' -> (i, k), or None."""
    return tuple(map(int, spec.split('/'))) if spec else None


def score(result, ref, downbeats=None):
    est = result.beats
    ref_t, est_t = mir_eval.beat.trim_beats(ref), mir_eval.beat.trim_beats(est)
    m = dict(f=mir_eval.beat.f_measure(ref_t, est_t))
    _, m['cmlt'], _, m['amlt'] = mir_eval.beat.continuity(ref_t, est_t)

    err = est - ref[np.abs(est[:, None] - ref[None, :]).argmin(1)] if len(est) else np.empty(0)
    hit = np.abs(err) < WINDOW
    m['offset_ms'] = 1000 * err[hit].mean() if hit.any() else np.nan
    m['spread_ms'] = 1000 * err[hit].std() if hit.any() else np.nan
    run = np.convolve(hit, np.ones(4))[3:len(hit)] == 4 if len(hit) else hit  # first of 4 consecutive hits
    m['lock_s'] = est[run.argmax()] - ref[0] if run.any() else np.nan
    later = ref[ref >= 5]
    m['locked'] = result.lock[np.searchsorted(result.times, later).clip(0, len(result.times) - 1)].mean()
    m['down_f'] = (mir_eval.beat.f_measure(mir_eval.beat.trim_beats(downbeats), mir_eval.beat.trim_beats(result.bars))
                   if downbeats is not None and len(downbeats) else np.nan)
    return m


def score_file(cfg, engine, audio, ann):
    x, sr = load_audio(audio)
    meter = load_meter(ann)  # the module's meter is set by hand, so score bars with the right one
    cfg = replace(cfg, beats_per_bar=meter) if meter else cfg
    return score((detect_cpp if engine == 'cpp' else detect)(x, sr, cfg), load_beats(ann), load_downbeats(ann))


def evaluate(root, cfg, annotations=None, out=None, jobs=None, engine='py', fold=None):
    found = list(pairs(root, annotations))
    if fold:  # only the files `beatref train --fold i/k` held out
        found = [p for p in found if fold_of(p[0], fold[1]) == fold[0]]
    if not found:
        raise SystemExit(f'no annotated audio found under {root}')
    rows = []
    with ProcessPoolExecutor(jobs) as pool:
        for (audio, _), m in zip(found, pool.map(score_file, repeat(cfg), repeat(engine), *zip(*found))):
            rows.append(dict(file=audio.name, **m))
            print(_fmt(audio.stem, m), flush=True)
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != 'file'}
    print('-' * 98)
    print(_fmt(f'mean of {len(rows)}', mean))
    if out:
        Path(out).write_text(json.dumps(dict(config=asdict(cfg), mean=mean, files=rows), indent=1, default=float))
    return mean, rows


def _fmt(name, m):
    return (f'{name[:32]:32s}  F {m["f"]:.3f}  CMLt {m["cmlt"]:.3f}  AMLt {m["amlt"]:.3f}  '
            f'offset {m["offset_ms"]:+6.1f}±{m["spread_ms"]:4.1f} ms  lock {m["lock_s"]:5.1f} s  '
            f'locked {m["locked"]:.0%}  bar F {m["down_f"]:.3f}')
