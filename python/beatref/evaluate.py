"""Evaluation: standard beat metrics (mir_eval) plus the ones that matter for a live clock."""
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import repeat
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from .detector import detect

AUDIO = {'.wav', '.flac', '.aiff', '.aif', '.au', '.ogg', '.mp3'}
WINDOW = 0.07  # match tolerance [s], as in mir_eval's F-measure


def load_audio(path):
    x, sr = sf.read(path, always_2d=True)
    return x.mean(1), sr


def load_beats(path):
    return np.loadtxt(path, ndmin=2)[:, 0]


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


def score(result, ref):
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
    return m


def score_file(cfg, audio, ann):
    x, sr = load_audio(audio)
    return score(detect(x, sr, cfg), load_beats(ann))


def evaluate(root, cfg, annotations=None, out=None, jobs=None):
    found = list(pairs(root, annotations))
    if not found:
        raise SystemExit(f'no annotated audio found under {root}')
    rows = []
    with ProcessPoolExecutor(jobs) as pool:
        for (audio, _), m in zip(found, pool.map(score_file, repeat(cfg), *zip(*found))):
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
            f'locked {m["locked"]:.0%}')
