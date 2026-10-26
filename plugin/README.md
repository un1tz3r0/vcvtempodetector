# Beat Detector (VCV Rack module)

Stage 1 of the plan: the Stage 0 reference (`python/beatref`) ported to C++ and wrapped in a
28HP Rack module. It listens to audio and outputs a predictive beat clock, with a scrolling
display of the band-coloured envelope, detected onsets and the beat grid.

## Build

```sh
make RACK_DIR=~/Rack-SDK            # plugin.so / .dylib / .dll
make RACK_DIR=~/Rack-SDK install    # into your Rack user folder (needs zstd)
make beattrack                      # offline driver for python/beatref; no SDK needed
```

Built against Rack SDK 2.6.6. CI builds Linux x64 (glibc 2.28 baseline, in a manylinux container),
Windows x64 (MinGW cross-compile, statically linked runtime) and macOS x64 and arm64. The code is C++17 (the Makefile overrides the SDK's
C++11 default).

## Panel

| Control | What it does |
| --- | --- |
| Follow | How fast the oscillator absorbs tempo and phase corrections: 0% rigid (2 s), 50% (0.2 s), 100% loose (0.02 s) |
| Phase, Time | Output offsets, ±180° of a beat and ±250 ms. Positive fires earlier, to cancel downstream delay |
| Ratio | Clock ×/÷ output: ÷8, ÷6, ÷4, ÷3, ÷2, ×2/3, ×1, ×3/2, ×2, ×3, ×4, ×6, ×8. Counted from bar 1 |
| Beats/bar | 1 to 16 |
| Tap | Each tap pulls the phase; two or more (under 2 s apart) seed the tempo. Flashes on every beat |
| ÷2, ×2 | Halve or double the tempo hypothesis |
| Nudge −, + | While held, shift the phase later or earlier (0.25 beat/s by default) |
| Bar reset | The nearest beat becomes bar 1 |
| Reset | Clear the tracker and start a fresh (or warm) acquisition. The outputs keep coasting meanwhile |

Tap, ÷2 and ×2 also pin the metrical level: until the next Reset, the tracker follows tempo
changes of about ±25% but can't drift back to double or half tempo.

Inputs: Audio L/mono and R (mono if R is unpatched, averaged otherwise; 5 V is full scale),
Clock, Tap, Reset, Bar reset (triggers) and Hold (gate: tracking freezes, the oscillator coasts).

Outputs: Beat, Bar and Clock ×/÷ triggers; Beat phase and Bar phase saws (0–10 V); BPM CV;
Lock gate; Confidence (0–10 V).

## Context menu

Tempo range (60–200 BPM) and preferred range (90–180), onset detection function (SuperFlux,
spectral flux, complex domain, HFC), BPM CV scaling (V/oct with 0 V = 120 BPM, 10 V = 300 BPM,
or 1 V = 100 BPM), warm start (off, a set tempo, or seeded by the first two Clock beats), what
to do on losing the beat (coast, stop after N bars, keep following) and the re-lock glide, lock
threshold, clock pull and pulses per beat, tap pull and nudge speed, trigger width, the
experimental automatic downbeat guess, display zoom and colours. An Advanced submenu exposes the
tracker's tuning (onset latency, low band weight, activation gamma, transition and observation
λ, preferred range strength, confidence smoothing and scale, silence threshold) with a button to
restore the defaults.

## Code

`src/dsp/` is the engine and has no Rack dependency: `OnsetDetector` (framing, FFT, the four
ODFs), `BeatTracker` (the online HMM), `Oscillator` (the flywheel) and `Engine` (activation
normalisation, confidence and lock, the audio-rate oscillator, bars, the clock divider, and the
manual controls). `BeatDetector.cpp` is the module and panel; `Waterfall.hpp` is the display. The
audio thread pushes one summary per analysis frame into a lock-free ring buffer, and the display
draws from that on the UI thread.

## Verification

```sh
make beattrack
beatref compare data/synth                     # C++ vs Python, beat for beat
beatref --engine cpp eval data/ballroom --annotations data/BallroomAnnotations
```

Results (2026-10-26):

- **Parity.** All 24 synthetic clips (and 8 more rendered at 48 kHz) produce the same beats as
  the reference, every one within a single sample (0.023 ms). That holds for all four ODFs and
  for the stop, follow and offset settings. Tempo and confidence agree to float printing
  precision, and the lock gate never differs.
- **Ballroom** through the C++ engine: F 0.769, CMLt 0.574, AMLt 0.789, offset −3.2 ± 18.7 ms,
  lock 4.7 s, locked 89%. These are identical to Stage 0.
- **CPU.** Offline, with the plugin's compiler flags, the engine uses 0.54% of a core at
  44.1 kHz and 0.78% at 48 kHz, where the analysis window is 4096 samples rather than 2048. Rack's
  per-module meter shows 0.9–1.3% in a slow cloud VM, where Fundamental's LFO reads 0.2%.
