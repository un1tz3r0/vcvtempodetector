from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    """Every tunable of the reference tracker. Mirrors the module's panel and context menu."""

    # tempo
    min_bpm: float = 60.0
    max_bpm: float = 200.0
    pref_min_bpm: float = 90.0
    pref_max_bpm: float = 180.0
    tempo_prior: float = 0.002        # per-frame log penalty per octave outside the preferred range

    # onset detection
    odf: str = 'superflux'            # superflux | flux | complex | hfc
    fps: float = 100.0
    latency: float = 0.013            # ODF peak delay behind the true onset [s], calibrated with `eval`
    model: str = ''                   # RTNeural JSON beat network; replaces the ODF activation (its latency is in the file)

    # activation normalisation
    norm_mean_tau: float = 1.0        # running mean [s]
    norm_peak_tau: float = 3.0        # peak follower release [s]
    norm_floor: float = 0.1           # peak never drops below this fraction of the long-term peak
    low_weight: float = 2.0           # weight of the low band (kick, bass) against mid and high
    act_gamma: float = 0.75           # compresses activations so weak but regular onsets still count
    act_floor: float = 0.01           # activations are clipped to [act_floor, 1 - act_floor]
    silence_db: float = -60.0         # below this RMS the activation is held at act_floor

    # HMM (madmom DBNBeatTrackingProcessor defaults)
    transition_lambda: float = 100.0
    observation_lambda: float = 16.0
    restart: float = 0.0              # per-frame probability of jumping anywhere: escapes wrong octaves, follows song changes

    # confidence and lock
    conf_tau: float = 2.0             # evidence smoothing [s]
    conf_scale: float = 2.0           # evidence [nats/s] that maps to confidence 0.73
    lock_on: float = 0.6
    lock_off: float = 0.4

    # oscillator
    follow: float = 0.5               # correction time constant: 0 rigid (2 s), 0.5 (0.2 s), 1 loose (0.02 s)
    glide_beats: float = 1.0
    lost: str = 'coast'               # coast | stop | follow
    stop_bars: int = 4
    phase_offset: float = 0.0         # degrees of a beat; positive fires earlier
    time_offset: float = 0.0          # seconds; positive fires earlier, to cancel downstream delay
    beats_per_bar: int = 4
    auto_downbeat: bool = False       # bar 1 follows the downbeat activation (network) or the low band (ODF)
    ratio: float = 2.0                # Clock x/÷ output, in clocks per beat

    # seeding
    warm_bpm: float = 0.0             # 0 = cold start
    warm_phase: float = 0.0
    clock_pull: float = 0.5           # 1 = the clock takes over and audio is ignored
    tap_pull: float = 0.9

    def override(self, pairs):
        """Apply 'name=value' strings, converting each value to the field's type."""
        types = {f.name: f.type for f in fields(self)}
        kw = {}
        for pair in pairs:
            name, value = pair.split('=', 1)
            if name not in types:
                raise KeyError(f'unknown setting {name!r}; choose from {", ".join(types)}')
            kw[name] = value.lower() in ('1', 'true', 'yes', 'on') if types[name] is bool else types[name](value)
        return replace(self, **kw)
