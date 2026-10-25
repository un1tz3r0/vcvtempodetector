# beatref

The Stage 0 reference for the Rack module: the same causal pipeline the C++ will run, in numpy,
plus an evaluation harness. Every setting lives in `beatref.config.Config`, which mirrors the
module's panel and context menu.

```sh
pip install -e '.[plot]'
beatref synth data/synth                                     # render the synthetic test set
beatref eval data/synth                                      # score it
beatref eval data/ballroom --annotations data/BallroomAnnotations --json ballroom.json
beatref track song.wav --out song.beats                      # beat times and beat-in-bar
beatref plot song.wav --ref song.beats --dur 20              # preview of the module display
beatref --set follow=0.8 --set odf=flux eval data/synth      # override any Config setting
```

## Pipeline

1. **Onset detection** (`odf.py`). Causal framing: frame *i* ends at sample (*i*+1)·hop, so
   nothing depends on the future. 100 frames/s, a 2048-sample Hann window at 44.1/48 kHz (the
   window scales with the sample rate rather than resampling). SuperFlux by default: a
   24 bands/octave log filterbank, `log10(1 + x)`, and a positive difference against a
   3-band maximum filter of the previous frame. Spectral flux, rectified complex domain and HFC
   (rectified rise) are selectable. The filterbank ODFs are summed separately for the low
   (< 200 Hz), mid and high (> 2 kHz) bands.
2. **Activation**. Each band is normalised by its own running mean (1 s) and peak follower
   (3 s release), so a kick counts as much as a hi-hat. The low band is weighted ×2, the
   result is compressed (γ = 0.75), and the activation is held at its floor below −60 dBFS.
3. **Tracker** (`tracker.py`). A port of madmom's `DBNBeatTrackingProcessor` in forward mode:
   the Krebs/Böck/Widmer (2015) state space, exponential tempo transitions (λ = 100) and a
   1/16-beat observation region. Additions: a tempo prior outside the preferred range; Tap and
   Clock beats as extra observations, delayed to line up with the activation; warm start; and
   an *evidence* signal. Evidence is the log-likelihood ratio of each frame under the tracker's
   prediction against a beat-agnostic null model. It is positive while the music keeps landing
   where beats are expected.
4. **Confidence and lock**. Evidence is smoothed over 2 s and passed through a logistic
   function. Lock uses hysteresis (on at 0.6, off at 0.4) and starts low, so the first lock has
   to be earned.
5. **Oscillator** (`oscillator.py`). An unwrapped beat-phase accumulator: beat *n* fires
   exactly once, so backward corrections never double-trigger. Phase and tempo errors are
   absorbed with a time constant set by Follow (2 s → 0.2 s → 0.02 s). A re-lock glides over
   `glide_beats`. While unlocked it coasts (or stops after N bars, or keeps following). Hold
   coasts without steering. Phase and time offsets are applied at the output; positive values
   fire earlier.

## Results

Defaults, 2026-10-25. F, CMLt and AMLt come from mir_eval, with the first 5 s trimmed. Offset is
the signed error of matched beats (negative means early). Lock is the time to the first of four
consecutive correct beats. Locked is the share of beats after 5 s with the Lock gate high.

| Set | Clips | F | CMLt | AMLt | Offset (ms) | Lock (s) | Locked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Ballroom](https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) ([annotations](https://github.com/CPJKU/BallroomAnnotations)) | 698 | 0.769 | 0.574 | 0.789 | −3.2 ± 18.7 | 4.7 | 89% |
| Synthetic (`beatref synth`, seed 1) | 24 | 0.752 | 0.640 | 0.876 | −7.2 ± 8.5 | 3.2 | 90% |

Ballroom F-measure for the other onset detection functions: spectral flux 0.726, complex
domain 0.568, HFC 0.568. With no music the lock stays off: silence is locked 0% of the time,
white noise 9%, and a sustained pad 3%.

Remaining failure modes, which the neural stage should address:

- **Metrical level.** Slow material (80–95 BPM) is often tracked at double tempo, and the
  transition model cannot leave the wrong level once it's locked. The ×2/÷2 buttons cover this
  on the module. A `restart` probability exists, but it hurt scores at every value tried.
- **Off-beat lock.** This happens on very sparse, syncopated grooves, for example the
  synthetic `sparse` pattern: a kick on 1 and a snare on the "and" of 3.
- **Bars.** Bars count from the first beat. Downbeat detection is deferred to Stage 2, and the
  module has Bar Reset for now.

Tap and Clock events are quantised to frames here (about 5 ms of bias). The module will apply
them at audio rate.
