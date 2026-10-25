"""Flywheel oscillator: the clock every output is derived from.

Phase is kept unwrapped (in beats), so beat n fires exactly once when the output phase first
reaches n; corrections that move phase backwards can never double-trigger. The tracker only
steers it: phase and tempo errors are absorbed with a time constant set by Follow, re-locks glide
over `glide_beats`, and with no lock it coasts, stops, or keeps following, per `lost`.
"""
import numpy as np


class Oscillator:
    def __init__(self, cfg, dt):
        self.cfg, self.dt = cfg, dt
        tau = 2.0 * (0.02 / 2.0) ** cfg.follow
        self.k_phase, self.k_tempo = dt / tau, dt / (2 * tau)
        self.offset = cfg.phase_offset / 360
        self.phase = 0.0        # unwrapped beats
        self.bps = 0.0          # beats per second
        self.fired = 0          # last beat index emitted
        self.running = False
        self.gliding = False
        self.unlocked_for = 0.0  # beats since lock was lost

    def out(self, phase=None):
        return (self.phase if phase is None else phase) + self.offset + self.cfg.time_offset * self.bps

    def step(self, t0, bpm, phase, locked, hold=False):
        """Advance from time t0 by one frame, steered by the tracker's (bpm, beat phase at t0 + dt).
        Hold coasts without steering. Returns the times and indices of beats fired this frame."""
        cfg = self.cfg
        if not self.running:
            if locked and not hold:  # first lock snaps: nothing downstream depends on the phase yet
                self.bps = bpm / 60
                self.phase = np.floor(self.phase) + 1 + phase
                self.fired = int(np.floor(self.out()))
                self.running, self.gliding, self.unlocked_for = True, False, 0.0
            return [], []

        before = self.out()
        self.phase += self.bps * self.dt
        after = self.out()
        idx = list(range(self.fired + 1, int(np.floor(after)) + 1))
        times = [t0 + self.dt * float(np.clip((n - before) / (after - before), 0, 1)) for n in idx]
        if idx:
            self.fired = idx[-1]
        if hold:
            return times, idx

        if locked:
            if self.unlocked_for > 0:
                self.gliding = True
            self.unlocked_for = 0.0
        else:
            self.unlocked_for += self.bps * self.dt
            if cfg.lost == 'stop' and self.unlocked_for >= cfg.stop_bars * cfg.beats_per_bar:
                self.running = False
            if cfg.lost != 'follow':
                return times, idx

        err = (phase - self.phase + 0.5) % 1 - 0.5
        k_phase, k_tempo = self.k_phase, self.k_tempo
        if self.gliding:
            k_phase = k_tempo = max(self.k_phase, self.dt * self.bps / cfg.glide_beats)
            self.gliding = abs(err) > 0.02
        self.phase += k_phase * err
        self.bps += k_tempo * (bpm / 60 - self.bps)
        return times, idx
