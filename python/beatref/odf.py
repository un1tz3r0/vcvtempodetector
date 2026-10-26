"""Causal onset detection functions, computed frame by frame exactly as the module will.

Frame i covers the `win` samples ending at sample (i + 1) * hop, so its value is available at
time (i + 1) * hop / sr and never depends on the future. Signals are processed in chunks so
memory stays flat for long inputs; the last two spectra carry over between chunks.
"""
import numpy as np

KINDS = ('superflux', 'flux', 'complex', 'hfc')
BANDS = (200.0, 2000.0)  # display band edges [Hz]: low | mid | high


def frame_params(sr, fps):
    hop = round(sr / fps)
    win = 1 << int(np.ceil(np.log2(0.0464 * sr)))  # 2048 at 44.1 kHz, 4096 at 48 kHz
    return win, hop


def log_filterbank(win, sr, bands_per_octave=24, fmin=30.0, fmax=17000.0):
    """Area-normalised triangular filters on a log frequency axis (madmom's SuperFlux front end)."""
    nbins = win // 2 + 1
    fmax = min(fmax, sr / 2)
    k = np.arange(np.floor(bands_per_octave * np.log2(fmin / 440)),
                  np.ceil(bands_per_octave * np.log2(fmax / 440)) + 1)
    centres = np.unique(np.round(440 * 2 ** (k / bands_per_octave) * win / sr).astype(int).clip(1, nbins - 1))
    fb = np.zeros((nbins, len(centres) - 2))
    for j, (a, c, b) in enumerate(zip(centres, centres[1:], centres[2:])):
        fb[a:c + 1, j] = np.linspace(0, 1, c - a + 1)
        fb[c:b + 1, j] = np.linspace(1, 0, b - c + 1)
        fb[:, j] /= fb[:, j].sum()
    return fb


class OnsetDetector:
    def __init__(self, sr, kind='superflux', fps=100.0):
        if kind not in KINDS:
            raise ValueError(f'odf must be one of {KINDS}')
        self.sr, self.kind = sr, kind
        self.win, self.hop = frame_params(sr, fps)
        self.fps = sr / self.hop
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.win) / self.win)
        self.fb = log_filterbank(self.win, sr)
        freqs = np.fft.rfftfreq(self.win, 1 / sr)
        self.band = np.digitize(freqs, BANDS)
        # filters grouped into the same low / mid / high bands, so each can be normalised on its own
        self.groups = np.eye(3)[np.digitize(freqs @ self.fb, BANDS)]

    def __call__(self, x, chunk=2048):
        """Return per-frame (odf, envelope, band energies) for mono signal x. The filterbank ODFs
        (superflux, flux) have one column per band (low, mid, high); the others have one column."""
        x = np.concatenate([np.zeros(self.win - self.hop), np.asarray(x, float)])
        n = (len(x) - self.win) // self.hop + 1
        frames = np.lib.stride_tricks.sliding_window_view(x, self.win)[::self.hop][:n]
        tail = np.zeros((2, self.win // 2 + 1), complex)
        odf, env, bands = [], [], []
        for i in range(0, n, chunk):
            block = frames[i:i + chunk]
            spec = np.fft.rfft(block * self.window, axis=1)
            odf.append(self._odf(np.concatenate([tail, spec]))[2:])
            tail = np.concatenate([tail, spec])[-2:]
            env.append(np.sqrt(np.mean(block[:, -self.hop:] ** 2, axis=1)))
            power = np.abs(spec) ** 2
            bands.append(np.stack([power[:, self.band == b].sum(1) for b in range(3)], axis=1))
        return np.concatenate(odf), np.concatenate(env), np.concatenate(bands)

    def _odf(self, spec):
        mag = np.abs(spec)
        if self.kind in ('superflux', 'flux'):
            logspec = np.log10(1 + mag @ self.fb)
            prev = np.roll(logspec, 1, axis=0)
            ref = prev.copy()
            if self.kind == 'superflux':  # maximum filter across 3 bands suppresses vibrato
                ref[:, 1:] = np.maximum(ref[:, 1:], prev[:, :-1])
                ref[:, :-1] = np.maximum(ref[:, :-1], prev[:, 1:])
            return np.clip(logspec - ref, 0, None) @ self.groups
        if self.kind == 'complex':  # rectified complex domain (Dixon 2006)
            m1, a1, a2 = np.roll(mag, 1, 0), np.angle(np.roll(spec, 1, 0)), np.angle(np.roll(spec, 2, 0))
            dev = np.abs(spec - m1 * np.exp(1j * (2 * a1 - a2)))
            return np.where(mag > m1, dev, 0).sum(1, keepdims=True)
        hfc = (mag ** 2 * np.arange(mag.shape[1])).sum(1)  # rectified rise of high frequency content
        return np.clip(hfc - np.roll(hfc, 1), 0, None)[:, None]
