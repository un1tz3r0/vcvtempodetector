"""Causal spectral features for the neural activation (Stage 2), the network's input.

The same framing as odf.py (frame i ends at sample (i + 1) * hop, 100 frames/s), but the window is
fixed in time rather than rounded to a power of two: 46.4 ms (2048 samples at 44.1 kHz), Hann,
zero-padded to the next power of two. The spectrum is interpolated at the bin frequencies of the
44.1 kHz, 2048-point grid and filtered there with madmom's log filterbank, so every sample rate
yields the same bands. Per frame: log10(1 + mul * band magnitude) and its positive
difference to the previous frame.
"""
from dataclasses import asdict, dataclass

import numpy as np

REF_SR, REF_WIN = 44100, 2048


@dataclass(frozen=True)
class Frontend:
    bands_per_octave: int = 12
    fmin: float = 30.0
    fmax: float = 17000.0
    log_mul: float = 1e4
    diff: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)

    def triangles(self):
        """(left, centre, right) bins of each filter on the 44.1 kHz / 2048-point grid (madmom's)."""
        nbins, bpo = REF_WIN // 2 + 1, self.bands_per_octave
        k = np.arange(np.floor(bpo * np.log2(self.fmin / 440)), np.ceil(bpo * np.log2(self.fmax / 440)) + 1)
        c = np.unique(np.round(440 * 2 ** (k / bpo) * REF_WIN / REF_SR).astype(int).clip(1, nbins - 1))
        return np.stack([c[:-2], c[1:-1], c[2:]], axis=1)

    @property
    def bands(self):
        return len(self.triangles())

    @property
    def size(self):
        return self.bands * (2 if self.diff else 1)


def frame_params(sr, fps=100.0):
    win = int(round(REF_WIN * sr / REF_SR))
    return win, 1 << int(np.ceil(np.log2(win))), int(round(sr / fps))


def filterbank(fe, sr, nfft):
    """The 44.1 kHz / 2048-point filterbank, applied to the spectrum on the `nfft`-point grid at `sr`
    linearly interpolated at the reference bin frequencies. One column per band."""
    nref = REF_WIN // 2 + 1
    ref = np.zeros((nref, fe.bands))
    k = np.arange(nref)
    for j, (a, c, b) in enumerate(fe.triangles()):
        ref[:, j] = np.clip(np.minimum((k - a) / (c - a), (b - k) / (b - c)), 0, None)
        ref[:, j] /= ref[:, j].sum()
    pos = k * REF_SR / REF_WIN * nfft / sr  # reference bin frequencies on this grid
    interp = np.zeros((nfft // 2 + 1, nref))
    inside = pos <= nfft // 2
    lo = np.floor(pos[inside]).astype(int)
    frac = pos[inside] - lo
    interp[lo, k[inside]] += 1 - frac
    interp[np.minimum(lo + 1, nfft // 2), k[inside]] += frac
    return interp @ ref


class Features:
    def __init__(self, sr, frontend=Frontend(), fps=100.0):
        self.sr, self.fe = sr, frontend
        self.win, self.nfft, self.hop = frame_params(sr, fps)
        self.fps = sr / self.hop
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.win) / self.win)
        self.window = w * 2 / w.sum()  # a full-scale sinusoid has magnitude 1
        self.fb = filterbank(frontend, sr, self.nfft)

    def bands(self, x, hop=None, chunk=4096):
        """Filtered magnitudes per frame, shape (frames, bands). `hop` (fractional samples) overrides
        the frame step, which time-stretches the sequence for training."""
        hop = hop or self.hop
        x = np.concatenate([np.zeros(self.win), np.asarray(x, float)])
        ends = np.round(np.arange(1, (len(x) - self.win) // hop + 1) * hop).astype(int) + self.win
        out = []
        for i in range(0, len(ends), chunk):
            idx = ends[i:i + chunk, None] - self.win + np.arange(self.win)
            out.append(np.abs(np.fft.rfft(x[idx] * self.window, self.nfft, axis=1)) @ self.fb)
        return np.concatenate(out) if out else np.zeros((0, self.fe.bands))

    def __call__(self, x):
        return transform(self.bands(x), self.fe)


def transform(bands, fe, gain=1.0):
    """Network input from filtered magnitudes: log bands, then their positive difference."""
    log = np.log10(1 + fe.log_mul * gain * bands)
    if not fe.diff:
        return log
    return np.concatenate([log, np.clip(np.diff(log, axis=0, prepend=0), 0, None)], axis=1)
