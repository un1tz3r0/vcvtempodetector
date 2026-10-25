"""Synthetic test set: drum, bass and pad arrangements with exact beat annotations.

Covers what a Rack patch throws at the module: steady sequenced grooves, tempo ramps and jumps,
breakdowns where the drums drop out, syncopation, swing and human timing jitter.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# onset positions in beats within a 4/4 bar
STYLES = {
    'four':      dict(kick=[0, 1, 2, 3], snare=[1, 3], hat=[.5, 1.5, 2.5, 3.5], bass=[.5, 1.5, 2.5, 3.5]),
    'breakbeat': dict(kick=[0, 1.75, 2.5], snare=[1, 3], hat=list(np.arange(0, 4, .5)), bass=[0, 1.75, 2.5]),
    'halftime':  dict(kick=[0, 2.75], snare=[2], hat=list(np.arange(0, 4, .5)), bass=[0, 2.75]),
    'sparse':    dict(kick=[0], snare=[2.5], hat=[1, 3], bass=[0, 1.5]),
}
MODES = ('steady', 'ramp', 'jump', 'breakdown')


@dataclass
class Scenario:
    name: str
    style: str
    mode: str
    bpm: float
    bars: int = 32
    swing: float = 0.0     # fraction of an eighth by which off-beats are late
    jitter: float = 0.0    # timing noise [s]
    seed: int = 0


def _env(sr, dur, tau):
    t = np.arange(int(dur * sr)) / sr
    return t, np.exp(-t / tau)


def _kick(sr, rng):
    t, e = _env(sr, .35, .12)
    return np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-t / .04)) / sr) * e


def _snare(sr, rng):
    t, e = _env(sr, .25, .07)
    noise = lfilter(*butter(2, [1500, 7000], 'band', fs=sr), rng.standard_normal(len(t)))
    return (.8 * noise + .5 * np.sin(2 * np.pi * 185 * t)) * e


def _hat(sr, rng):
    t, e = _env(sr, .08, .018)
    return .35 * lfilter(*butter(2, 7000, 'high', fs=sr), rng.standard_normal(len(t))) * e


def _bass(sr, rng, f):
    t, e = _env(sr, .4, .15)
    return .5 * lfilter(*butter(1, 600, fs=sr), 2 * ((f * t) % 1) - 1) * e


def beat_times(sc):
    """Time of every beat (plus one past the end) for the scenario's tempo curve."""
    n = sc.bars * 4
    beats = np.arange(n + 1)
    if sc.mode == 'ramp':
        bpm = np.interp(beats, [0, n // 4, 3 * n // 4, n], [sc.bpm, sc.bpm, sc.bpm * 1.12, sc.bpm * 1.12])
    elif sc.mode == 'jump':
        bpm = np.where(beats < n // 2, sc.bpm, sc.bpm * 1.25)
    else:
        bpm = np.full(n + 1, sc.bpm)
    return np.r_[0, np.cumsum(60 / bpm[:-1])] + 1.0


def render(sc, sr=44100):
    """Return (audio, beat times, beat-in-bar numbers)."""
    rng = np.random.default_rng(sc.seed)
    beats = beat_times(sc)
    out = np.zeros(int((beats[-1] + 1) * sr))
    drop = range(sc.bars // 2, sc.bars // 2 + 4) if sc.mode == 'breakdown' else range(0)
    root = rng.choice([41.2, 43.65, 49.0, 55.0])
    voices = dict(kick=_kick, snare=_snare, hat=_hat)

    def place(sig, t, gain):
        i = int(t * sr)
        seg = out[i:i + len(sig)]
        seg += gain * sig[:len(seg)]

    for bar in range(sc.bars):
        for part, positions in STYLES[sc.style].items():
            if bar in drop and part != 'bass':
                continue
            for p in positions:
                b = bar * 4 + p
                if b % 1 == .5:
                    b += sc.swing * .5
                t = np.interp(b, np.arange(len(beats)), beats) + rng.normal(0, sc.jitter)
                gain = rng.uniform(.8, 1.0)
                if part == 'bass':
                    place(_bass(sr, rng, root * rng.choice([1, 1, 1.5, 2])), t, gain)
                else:
                    place(voices[part](sr, rng), t, gain)

    t = np.arange(len(out)) / sr  # sustained pad keeps harmony going through breakdowns
    pad = sum(np.sin(2 * np.pi * root * 4 * r * t + rng.uniform(0, 6)) for r in (1, 1.26, 1.5))
    out += .04 * pad * np.minimum(t / 2, 1) + 1e-3 * rng.standard_normal(len(out))
    out *= .9 / np.abs(out).max()
    return out, beats[:-1], np.arange(len(beats) - 1) % 4 + 1


def scenarios(count=24, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        style, mode = list(STYLES)[i % len(STYLES)], MODES[(i // len(STYLES)) % len(MODES)]
        human = i >= len(STYLES) * len(MODES)
        out.append(Scenario(f'{i:02d}-{style}-{mode}{"-human" if human else ""}', style, mode,
                            round(float(rng.uniform(75, 175)), 1),
                            swing=float(rng.uniform(.1, .3)) if human else 0.0,
                            jitter=.008 if human else 0.0, seed=int(rng.integers(1 << 31))))
    return out


def write_set(root, count=24, seed=1, sr=44100):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for sc in scenarios(count, seed):
        audio, beats, numbers = render(sc, sr)
        sf.write(root / f'{sc.name}.wav', audio, sr, subtype='PCM_16')
        np.savetxt(root / f'{sc.name}.beats', np.c_[beats, numbers], fmt=['%.4f', '%d'], delimiter='\t')
        print(f'{sc.name:28s} {sc.bpm:6.1f} BPM  {len(audio) / sr:5.1f} s')
