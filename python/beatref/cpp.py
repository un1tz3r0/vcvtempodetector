"""Run the module's C++ engine (plugin/build/beattrack) on the same input as the reference."""
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Config
from .detector import Result

DEFAULT = Path(__file__).resolve().parents[2] / 'plugin' / 'build' / 'beattrack'


def binary():
    path = Path(os.environ.get('BEATTRACK', DEFAULT))
    if not path.exists():
        raise SystemExit(f'{path} not found: run `make beattrack` in plugin/, or set BEATTRACK')
    return path


def detect_cpp(x, sr, cfg=Config()):
    """The C++ counterpart of detector.detect (no taps, clock or hold)."""
    args = [str(binary()), '--sr', str(sr)]
    for name, value in asdict(cfg).items():
        args += ['--set', f'{name}={value}']
    with tempfile.NamedTemporaryFile(suffix='.tsv') as frames:
        out = subprocess.run(args + ['--frames', frames.name], input=np.asarray(x, '<f8').tobytes(),
                             capture_output=True, check=True).stdout.decode()
        f = np.loadtxt(frames.name, ndmin=2)
    b = np.loadtxt(out.splitlines(), ndmin=2) if out.strip() else np.empty((0, 2))
    beats, bar = b[:, 0], b[:, 1] == 1
    n = len(f)
    return Result(cfg, n / f[-1, 0] if n else cfg.fps, f[:, 0], f[:, 1:4], f[:, 4], np.zeros(n), np.zeros(n), np.zeros((n, 3)),
                  f[:, 5], f[:, 6], f[:, 7], f[:, 8].astype(bool), beats, np.arange(len(beats)), bar)
