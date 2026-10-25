"""The whole chain: onset detection, activation normalisation, tracker, oscillator."""
from dataclasses import dataclass

import numpy as np

from .config import Config
from .odf import OnsetDetector
from .oscillator import Oscillator
from .tracker import BeatTracker


@dataclass
class Result:
    cfg: Config
    fps: float
    times: np.ndarray   # frame times [s]: when each frame's values become available
    odf: np.ndarray
    act: np.ndarray     # normalised beat activation
    env: np.ndarray     # RMS envelope, for display
    bands: np.ndarray   # low / mid / high energy, for display
    bpm: np.ndarray
    phase: np.ndarray   # tracker beat phase, latency compensated
    conf: np.ndarray    # confidence: logistic of the smoothed tracker evidence
    lock: np.ndarray
    beats: np.ndarray   # output beat times [s]
    index: np.ndarray   # unwrapped beat numbers

    @property
    def bars(self):
        return self.beats[(self.index - self.index[:1]) % self.cfg.beats_per_bar == 0]

    def clock(self, ratio=None):
        """Clock x/÷ output times: `ratio` clocks per beat, interpolated along the beat grid."""
        ratio = ratio or self.cfg.ratio
        if len(self.beats) < 2:
            return np.empty(0)
        n = np.arange(np.ceil(self.index[0] * ratio), np.floor(self.index[-1] * ratio) + 1) / ratio
        return np.interp(n, self.index, self.beats)


def detect(x, sr, cfg=Config(), taps=(), clock=(), hold=()):
    """Track mono signal x. `taps` and `clock` are event times [s]; `hold` is (start, end) spans."""
    odf, env, bands = (od := OnsetDetector(sr, cfg.odf, cfg.fps))(x)
    fps, dt, n = od.fps, 1 / od.fps, len(odf)
    times = (np.arange(n) + 1) * dt

    # Tap and Clock beats are true beat times; delay them to line up with the lagging activation
    events = np.zeros(n)
    for ts, pull in ((taps, cfg.tap_pull), (clock, cfg.clock_pull)):
        k = np.floor(np.asarray(ts, float) * fps).astype(int) + round(cfg.latency * fps)
        k = k[(k >= 0) & (k < n)]
        events[k] = np.maximum(events[k], pull)
    held = np.zeros(n, bool)
    for a, b in hold:
        held |= (times >= a) & (times < b)
    deaf = len(clock) > 0 and cfg.clock_pull >= 1

    trk, osc = BeatTracker(cfg, fps), Oscillator(cfg, dt)
    k_mean, k_conf = dt / cfg.norm_mean_tau, dt / cfg.conf_tau
    r_peak, r_long = np.exp(-dt / cfg.norm_peak_tau), np.exp(-dt / 20)
    mean = peak = long = np.zeros(odf.shape[1])
    evidence, quiet = -2 * cfg.conf_scale, 10 ** (cfg.silence_db / 20)  # start unconfident: lock must be earned
    weights = np.r_[cfg.low_weight, 1.0, 1.0][:odf.shape[1]]
    weights /= weights.sum()
    locked = False
    act, bpm, phase, cf, lock = (np.zeros(n) for _ in range(5))
    beats, index = [], []
    for i, o in enumerate(odf):
        # each band normalised by its own running mean and peak, so kicks count at least as much as hats
        mean = mean + k_mean * (o - mean)
        long = np.maximum(o, long * r_long)
        peak = np.maximum.reduce([o, peak * r_peak, cfg.norm_floor * long])
        a = (np.clip((o - mean) / (peak - mean + 1e-12), 0, 1) @ weights) ** cfg.act_gamma
        act[i] = np.clip(a if env[i] > quiet else 0, cfg.act_floor, 1 - cfg.act_floor)

        bpm[i], ph, ev = trk.step(None if held[i] or deaf else act[i], events[i])
        phase[i] = (ph + cfg.latency * bpm[i] / 60) % 1
        evidence += k_conf * (ev * fps - evidence)  # nats per second
        cf[i] = 1 / (1 + np.exp(-evidence / cfg.conf_scale))
        locked = cf[i] > (cfg.lock_off if locked else cfg.lock_on)
        lock[i] = locked

        t, k = osc.step(times[i], bpm[i], phase[i] + bpm[i] / 60 * dt, locked, held[i])
        beats += t
        index += k
    return Result(cfg, fps, times, odf, act, env, bands, bpm, phase, cf, lock.astype(bool),
                  np.array(beats), np.array(index, int))
