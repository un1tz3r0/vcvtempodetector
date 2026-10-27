"""Beat-for-beat comparison of the C++ engine against the reference (Stage 1 exit criterion)."""
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

from .cpp import detect_cpp
from .detector import detect
from .evaluate import AUDIO, load_audio


def compare_file(cfg, path):
    x, sr = load_audio(path)
    p, c = detect(x, sr, cfg), detect_cpp(x, sr, cfg)
    keep = p.beats < len(x) / sr  # the reference extrapolates one frame past the end of the input
    p.beats, p.bar = p.beats[keep], p.bar[keep]
    d = np.abs(p.beats[:, None] - c.beats[None, :]).min(1) if len(p.beats) and len(c.beats) else np.empty(0)
    return dict(py=len(p.beats), cpp=len(c.beats), max_ms=1000 * d.max() if len(d) else 0.0,
                bars=int((p.bar != c.bar).sum()) if len(p.bar) == len(c.bar) else -1,
                bpm=float(np.abs(p.bpm - c.bpm).max()), conf=float(np.abs(p.conf - c.conf).max()),
                lock=int((p.lock != c.lock).sum()))


def compare(root, cfg, jobs=None, tolerance_ms=1.0):
    files = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in AUDIO)
    worst, failed = 0.0, 0
    with ProcessPoolExecutor(jobs) as pool:
        for path, m in zip(files, pool.map(compare_file, repeat(cfg), files)):
            ok = m['py'] == m['cpp'] and m['max_ms'] <= tolerance_ms and m['bars'] == 0
            failed += not ok
            worst = max(worst, m['max_ms'])
            print(f'{path.stem[:32]:32s}  beats {m["py"]:4d} / {m["cpp"]:4d}  max Δ {m["max_ms"]:6.3f} ms  '
                  f'ΔBPM {m["bpm"]:.1e}  Δconf {m["conf"]:.1e}  lock frames differing {m["lock"]}  bar 1 differing {m["bars"]}'
                  f'{"" if ok else "  FAIL"}', flush=True)
    print(f'{len(files) - failed}/{len(files)} files within {tolerance_ms} ms, worst {worst:.3f} ms')
    return failed == 0
