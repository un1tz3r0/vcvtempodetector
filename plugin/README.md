# Beat Detector (VCV Rack module)

The `python/beatref` reference ported to C++ and wrapped in a 28HP Rack module. It listens to
audio and outputs a predictive beat clock, with a scrolling display of the band-coloured envelope,
detected onsets and the beat grid. Since 2.1 the beat activation comes from a small neural network
(Stage 2) by default; the onset detection function (Stage 1) is still one menu click away.

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

## Beat activation

**Neural network** (the default for new modules): a causal convolutional network, 110k
parameters, trained on Ballroom and GTZAN (see [`python/`](../python/#neural-activation-stage-2)).
It outputs a beat and a downbeat activation every 10 ms, and the tracker, lock and oscillator
work as before. It runs through [RTNeural](https://github.com/jatinchowdhury18/RTNeural) on its own
worker thread. The audio thread only copies samples into a lock-free ring and reads results back,
and it never waits or makes a system call. The tracker compensates for how many frames the network
is behind. If the worker falls more than 0.64 s behind, frames are dropped and the tracker coasts
over them.

**Load RTNeural JSON…** loads another model, for example one trained with `beatref train`. It
must be a sequential RTNeural model (dense, conv1d, GRU, LSTM, batchnorm, PReLU or activation
layers) whose input size matches the front end in its `beatref` block. It needs 1 output (beat), 2
(beat, downbeat) or 3 (softmax over none, beat, downbeat). A loaded model is saved in the patch,
so the patch works on other machines. A file that doesn't load leaves the current model in place
and shows the error in the menu. **Use the built-in model** goes back to the built-in model.

**Onset detection function** is the Stage 1 activation. Patches saved before 2.1 keep it.

## Learning mode

Adapts the network to your own material (Stage 3). Patch a trusted beat clock into **Clock**
(a sequencer or a clock module driving the music you feed in, with *Clock input → Pulses per
beat* set to match) and choose **Learning mode → Record**. The module writes the audio it hears
and the clock's beats to `<Rack user folder>/BeatDetector/recordings/<date>-<time>.wav` and
`.beats`, until you stop it, unpatch the clock or the sample rate changes. Pulses on **Bar reset**
mark bar 1: the nearest clock beat becomes bar 1 and the others are numbered with Beats/bar;
without a bar reset, only beat times are written. The audio thread only copies into a lock-free
ring (6 s deep); a worker thread writes the file, and the grid stays aligned with the audio that
was kept even if the disk stalls.

Adapt a model to the recordings with `beatref finetune` or the
[Colab notebook](../python/notebooks/learning_mode.ipynb) (see
[`python/`](../python/#learning-mode-stage-3)), then load it. **Model library** lists the
`.json` models in `<Rack user folder>/BeatDetector/models`, loads one with a click and saves the
current model there. A model adapted by `beatref finetune` carries the tempo range and meter of
its recordings, and **Apply the model's suggested settings** sets the tempo ranges and Beats/bar
to them.

## Context menu

Beat activation and learning mode (above), tempo range (60–200 BPM) and preferred range (90–180), onset detection function (SuperFlux,
spectral flux, complex domain, HFC), BPM CV scaling (V/oct with 0 V = 120 BPM, 10 V = 300 BPM,
or 1 V = 100 BPM), warm start (off, a set tempo, or seeded by the first two Clock beats), what
to do on losing the beat (coast, stop after N bars, keep following) and the re-lock glide, lock
threshold, clock pull and pulses per beat, tap pull and nudge speed, trigger width, automatic
downbeat (bar 1 follows the network's downbeat output, or the low band with the ODF), display
zoom and colours. An Advanced submenu exposes the
tracker's tuning (onset latency, low band weight, activation gamma, transition and observation
λ, preferred range strength, confidence smoothing and scale, silence threshold) with a button to
restore the defaults.

## Code

`src/dsp/` is the engine and has no Rack dependency: `OnsetDetector` (framing, FFT, the four
ODFs), `BeatTracker` (the online HMM), `Oscillator` (the flywheel) and `Engine` (activation
normalisation, confidence and lock, the audio-rate oscillator, bars, the clock divider, and the
manual controls), plus the neural path: `Features` (the network's filterbank front end),
`Network` (an RTNeural model and its `beatref` metadata) and `Inference` (the worker thread and
its rings), and `Recorder` (learning mode's WAV and grid writer). `BeatDetector.cpp` is the module and panel; `Waterfall.hpp` is the display. RTNeural
and nlohmann/json are vendored header-only in `vendor/` and run in double precision with the
plain C++ backend, so they build on every Rack platform, ARM included, and match the Python
reference exactly. A new model is built on the UI thread and handed to the audio thread through
an atomic mailbox; the old one is freed back on the UI thread. The
audio thread pushes one summary per analysis frame into a lock-free ring buffer, and the display
draws from that on the UI thread.

## Verification

```sh
make beattrack
beatref compare data/synth                     # C++ vs Python, beat for beat
beatref --set model=plugin/res/models/beat.json --set auto_downbeat=1 compare data/synth
beatref --engine cpp eval data/ballroom --annotations data/BallroomAnnotations
```

`beattrack --threaded` runs the network on the worker thread, as the module does, with the input
paced at 8× real time. Without it the network runs inline, which is deterministic.

Results (2026-10-27, neural):

- **Parity.** With the built-in model and automatic downbeat on, all 24 synthetic clips produce the
  same beats and the same bar 1 flags as the Python reference, every beat within 0.023 ms.
  Threaded runs give the same beats, within 3 ms, because results arrive a frame or two late and
  the lag compensation absorbs that.
- **Accuracy.** See the cross-validated results in [`python/`](../python/#results): Ballroom F
  0.856 against 0.769 for the ODF, and GTZAN F 0.835 against 0.749.
- **CPU.** Offline at 44.1 kHz, the network and its front end add 0.70% of a core (1.07% at
  48 kHz) on the worker thread. The audio thread's own cost goes up by about 0.2%. Rack's meter,
  which counts only the audio thread, reads 0.8–1.3% in a slow cloud VM, the same as with the
  ODF.
- **Live.** In Rack 2.6.6 the module locks to and holds a 120 BPM noise-burst pattern. These all
  work while it runs: switching activation, a user model restored from the patch, falling back
  from a broken model file, and a pre-2.1 patch (which keeps the ODF).
- **Context menu** (2.1.1). Hovering each custom item opens its submenu, nested ones included.
  Before 2.1.1 the submenu builders reached the module through a reference to a local of
  `appendContextMenu`, which dangles once the menu is shown, so hovering Tempo range crashed Rack.

Results (2026-10-26, ODF):

- **Parity.** All 24 synthetic clips (and 8 more rendered at 48 kHz) produce the same beats as
  the reference, every one within a single sample (0.023 ms). That holds for all four ODFs and
  for the stop, follow and offset settings. Tempo and confidence agree to float printing
  precision, and the lock gate never differs.
- **Ballroom** through the C++ engine: F 0.769, CMLt 0.574, AMLt 0.789, offset −3.2 ± 18.7 ms,
  lock 4.7 s, locked 89%. These are identical to Stage 0.
- **CPU.** Offline, with the plugin's compiler flags, the engine uses 0.54% of a core at
  44.1 kHz and 0.78% at 48 kHz, where the analysis window is 4096 samples rather than 2048. Rack's
  per-module meter shows 0.9–1.3% in a slow cloud VM, where Fundamental's LFO reads 0.2%.
