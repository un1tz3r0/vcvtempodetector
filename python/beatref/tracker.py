"""Online beat tracker: a port of madmom's DBNBeatTrackingProcessor in forward (online) mode.

State space and transitions follow Krebs, Böck & Widmer (2015): one chain of `interval` states
per tempo, advancing one state per frame; tempo may change only when a chain wraps, with an
exponential penalty on the interval ratio. The first 1/observation_lambda of each beat is the
beat region. Additions over madmom: a tempo prior favouring the preferred range, external beat
events (Tap, Clock) as extra observations, warm start, an optional restart probability, and a
per-frame evidence measure that drives confidence and lock.
"""
import numpy as np


class BeatTracker:
    def __init__(self, cfg, fps):
        self.fps = fps
        self.intervals = np.arange(round(60 * fps / cfg.max_bpm), round(60 * fps / cfg.min_bpm) + 1)
        self.first = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.last = np.cumsum(self.intervals) - 1
        self.tempo_of = np.repeat(np.arange(len(self.intervals)), self.intervals)
        self.pos = np.concatenate([np.arange(i) / i for i in self.intervals])
        self.phasor = np.exp(2j * np.pi * self.pos)

        ratio = self.intervals[None, :] / self.intervals[:, None]
        trans = np.exp(-cfg.transition_lambda * np.abs(ratio - 1))
        trans[trans <= np.spacing(1)] = 0
        self.trans = trans / trans.sum(1, keepdims=True)

        self.beat = self.pos < 1 / cfg.observation_lambda
        self.lam = cfg.observation_lambda
        self.restart = cfg.restart
        bpm = 60 * fps / self.intervals[self.tempo_of]
        outside = np.maximum(np.log2(cfg.pref_min_bpm / bpm), 0) + np.maximum(np.log2(bpm / cfg.pref_max_bpm), 0)
        self.prior = np.exp(-cfg.tempo_prior * outside)
        self.reset(cfg.warm_bpm, cfg.warm_phase)

    def reset(self, bpm=0.0, phase=0.0):
        """Uniform belief, or one concentrated near (bpm, phase) for a warm start."""
        if bpm > 0:
            interval = self.intervals[self.tempo_of]
            p = np.exp(-0.5 * (np.log2(60 * self.fps / interval / bpm) / 0.05) ** 2
                       + 8 * np.cos(2 * np.pi * (self.pos - phase)))
        else:
            p = np.ones(len(self.pos))
        self.p = p / p.sum()

    def step(self, act=None, event=0.0):
        """Advance one frame. `act` is the beat activation in (0, 1), or None for no audio evidence
        (Hold, or a clock at full pull). `event` > 0 is the pull strength of a Tap or Clock beat.
        Returns (bpm, phase in [0, 1), evidence): evidence is the log likelihood ratio of this
        frame's observations under the tracker's prediction against a beat-agnostic null model, so
        it is positive while the music keeps landing where the tracker expects beats."""
        p = self.p
        q = np.empty_like(p)
        q[1:] = p[:-1]
        q[self.first] = p[self.last] @ self.trans
        if self.restart:
            q = (1 - self.restart) * q + self.restart / len(q)
        on = q[self.beat].sum() / q.sum()  # predicted probability of being in a beat region
        evidence = 0.0
        if act is not None:
            beat, rest = act, (1 - act) / (self.lam - 1)
            q *= np.where(self.beat, beat, rest)
            evidence += np.log((on * beat + (1 - on) * rest) / (beat / self.lam + (1 - 1 / self.lam) * rest))
        if event > 0:
            w = 1 + (event / (1 - event) if event < 1 else 1e6)
            q[self.beat] *= w
            evidence += np.log((on * w + 1 - on) / (w / self.lam + 1 - 1 / self.lam))
        q *= self.prior
        self.p = q / q.sum()
        return (*self.estimate(), evidence)

    def estimate(self):
        """(bpm, phase): the mean interval within 3% of the most likely one, and the circular mean
        phase of all states."""
        m = np.bincount(self.tempo_of, self.p, len(self.intervals))
        near = np.abs(self.intervals / self.intervals[m.argmax()] - 1) <= 0.03
        interval = (m[near] * self.intervals[near]).sum() / m[near].sum()
        return 60 * self.fps / interval, (np.angle((self.p * self.phasor).sum()) / (2 * np.pi)) % 1
