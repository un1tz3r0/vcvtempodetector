"""The whole chain: onset detection, activation normalisation, tracker, oscillator."""
from dataclasses import dataclass, replace

import numpy as np

from .config import Config
from .network import Network
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
    down: np.ndarray    # downbeat activation (neural models with a downbeat output), else zeros
    env: np.ndarray     # RMS envelope, for display
    bands: np.ndarray   # low / mid / high energy, for display
    bpm: np.ndarray
    phase: np.ndarray   # tracker beat phase, latency compensated
    conf: np.ndarray    # confidence: logistic of the smoothed tracker evidence
    lock: np.ndarray
    beats: np.ndarray   # output beat times [s]
    index: np.ndarray   # unwrapped beat numbers
    bar: np.ndarray     # whether each beat was bar 1

    @property
    def bars(self):
        return self.beats[self.bar]

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
    net = cfg.model and Network.load(cfg.model)
    if net:  # the network's activation replaces the normalised ODF; its own latency replaces cfg.latency
        beat, down = net.activations(x, sr, cfg.fps)
        cfg = replace(cfg, latency=net.latency)
    else:
        down = None
    # downbeat evidence: the network's downbeat output, else the low band of a three-band ODF
    by_net, by_low = net and down is not None, not net and odf.shape[1] == 3
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
    beats, index, bar = [], [], []
    downbeat, origin, bpb = np.zeros(16), None, min(cfg.beats_per_bar, 16)
    for i, o in enumerate(odf):
        # each band normalised by its own running mean and peak, so kicks count at least as much as hats
        mean = mean + k_mean * (o - mean)
        long = np.maximum(o, long * r_long)
        peak = np.maximum.reduce([o, peak * r_peak, cfg.norm_floor * long])
        norm = np.clip((o - mean) / (peak - mean + 1e-12), 0, 1)
        a = beat[i] if net else (norm @ weights) ** cfg.act_gamma
        act[i] = np.clip(a if env[i] > quiet else 0, cfg.act_floor, 1 - cfg.act_floor)

        bpm[i], ph, ev = trk.step(None if held[i] or deaf else act[i], events[i])
        phase[i] = (ph + cfg.latency * bpm[i] / 60) % 1
        evidence += k_conf * (ev * fps - evidence)  # nats per second
        cf[i] = 1 / (1 + np.exp(-evidence / cfg.conf_scale))
        locked = cf[i] > (cfg.lock_off if locked else cfg.lock_on)
        lock[i] = locked

        t, k = osc.step(times[i], bpm[i], phase[i] + bpm[i] / 60 * dt, locked, held[i])

        # automatic downbeat: bar 1 is the beat position collecting the most downbeat evidence (the
        # network's downbeat activation, else the low band), decaying over 8 s
        downbeat *= np.exp(-dt / 8)
        if cfg.auto_downbeat and (by_net or by_low) and osc.running and origin is not None and bpb > 1:
            u = osc.out() - (dt + cfg.latency) * osc.bps  # the beat position this frame's activation describes
            if abs(u - round(u)) < 0.08:
                downbeat[int((round(u) - origin) % bpb)] += down[i] if by_net else norm[0]
            best = int(np.argmax(downbeat[:bpb]))
            if best and downbeat[best] > 1.25 * downbeat[0]:
                origin += best
                downbeat[:bpb] = np.roll(downbeat[:bpb], -best)
        if k and origin is None:
            origin = k[0]
        beats += t
        index += k
        bar += [(j - origin) % cfg.beats_per_bar == 0 for j in k]
    return Result(cfg, fps, times, odf, act, down if by_net else np.zeros(n), env, bands, bpm, phase, cf, lock.astype(bool),
                  np.array(beats), np.array(index, int), np.array(bar, bool))
