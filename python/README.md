# beatref

The reference for the Rack module: the same causal pipeline the C++ runs, in numpy, plus an
evaluation harness and the training code for the neural activation (Stage 2). Every setting
lives in `beatref.config.Config`, which mirrors the module's panel and context menu.

```sh
pip install -e '.[plot]'                                     # '.[plot,train]' adds torch for `train`
beatref synth data/synth                                     # render the synthetic test set
beatref eval data/synth                                      # score it
beatref eval data/ballroom --annotations data/BallroomAnnotations --json ballroom.json
beatref track song.wav --out song.beats                      # beat times and beat-in-bar
beatref plot song.wav --ref song.beats --dur 20              # preview of the module display
beatref --set follow=0.8 --set odf=flux eval data/synth      # override any Config setting
beatref compare data/synth                                   # the module's C++ engine against this, beat for beat
beatref --engine cpp eval data/synth                         # track or eval through the C++ engine
beatref --set model=../plugin/res/models/beat.json eval ...  # the neural activation instead of the ODF
beatref train data/ballroom:data/BallroomAnnotations data/gtzan:data/gtzan-beats --out beat.json
beatref finetune ../plugin/res/models/beat.json recordings/ --out mine.json  # learning mode
beatref --set model=f0.json eval data/ballroom --annotations data/BallroomAnnotations --fold 0/4
```

The C++ commands use `plugin/build/beattrack` (`make beattrack` in `plugin/`), or the binary
named by `$BEATTRACK`.

## Pipeline

1. **Onset detection** (`odf.py`). Causal framing: frame *i* ends at sample (*i*+1)·hop, so
   nothing depends on the future. 100 frames/s, and a Hann window of 46 ms rounded up to a power
   of two: 2048 samples at 44.1 kHz, 4096 at 48 kHz. It scales with the sample rate rather than
   resampling. SuperFlux by default: a 24 bands/octave log filterbank, `log10(1 + x)`, and a
   positive difference against a 3-band maximum filter of the previous frame. Spectral flux,
   rectified complex domain and HFC (rectified rise) are selectable. The filterbank ODFs are
   summed separately for the low (< 200 Hz), mid and high (> 2 kHz) bands.
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

## Neural activation (Stage 2)

With `model` set, a network's beat output replaces step 2's activation; the tracker,
confidence and oscillator are unchanged. Models are RTNeural JSON files (Keras layout) with a
`beatref` block that holds the front end, the outputs and the latency. `network.py` runs them
in numpy, and the module runs the same file through RTNeural.

- **Front end** (`features.py`). The same causal 100 frames/s framing and 46 ms Hann window,
  zero-padded to a power of two. The spectrum is interpolated at the 44.1 kHz / 2048 bin
  frequencies, so the bands are identical at every sample rate. Then madmom's 12 bands/octave
  filterbank (30 Hz to 17 kHz, 82 bands), `log10(1 + 10⁴·x)` with full scale at 1, and its
  positive difference: 164 inputs.
- **Network** (`train.py`). A dense layer (64, ELU), eight causal convolutions (kernel 3,
  dilations 1 to 128, ELU), and a dense layer with two sigmoid outputs: beat and downbeat.
  That gives 110k parameters and a 5.1 s receptive field, with nothing recurrent. The input
  standardisation is folded into the first layer. A GRU can be added with `--hyper gru=N`;
  on CPU it trained 2.5× slower and scored no better.
- **Training**. Ballroom and GTZAN, with targets one frame after the frame that contains the
  annotation and the two neighbours weighted ½. Time stretch is done by scaling the hop by
  0.8 to 1.25, which changes the tempo but not the timbre. Gain is random from −30 to +6 dB.
  The loss is binary cross-entropy, trained for 30 epochs of 250 batches of 15 s crops with
  AdamW and a one-cycle schedule. The output bias starts at the base-rate logit; from zero,
  training sits on a plateau for many epochs. The best epoch on 10% held-back files is
  exported, and its latency is then calibrated so that tracked beats on those files land on
  the annotations on average.
- **Folds**. Files are split by a hash of their name. `train --fold i/k` leaves out the files
  that `eval --fold i/k` scores.
- **Downbeats** (`auto_downbeat`). Each tracked beat's bar position is scored by the downbeat
  activation within ±0.08 beats of it. The scores decay with an 8 s time constant. Bar 1 moves
  when another position beats the current one by 25%. Without a downbeat output, the ODF's low
  band is used instead. Bar F is scored against the annotated meter, as if Beats/Bar were set
  to match.

## Learning mode (Stage 3)

`beatref finetune BASE.json SOURCES... --out ADAPTED.json` adapts a model to your own material:
audio with a beat grid aligned to it, in the same `time [beat-in-bar]` layout as the datasets.
The module's *Learning mode → Record* (from 2.2.0) writes exactly that from the Clock input, and the
[Colab notebook](notebooks/learning_mode.ipynb)
([open in Colab](https://colab.research.google.com/github/un1tz3r0/vcvtempodetector/blob/main/python/notebooks/learning_mode.ipynb))
runs it on recordings in Google Drive.

- **What is adapted.** The top `layers` layers (default 3: the output layer and the top two
  convolutions, 25k of 110k parameters; 0 adapts all of them), with an L2-SP penalty
  (`l2sp`) that pulls the weights back towards the base model rather than towards zero. The
  same stretch and gain augmentation as training; `--replay ROOT[:ANN]` with
  `--tune replay=0.5` mixes the base model's training data into half the batches.
- **Validation.** The last 20% of every recording (`holdout`) is never trained on. The base
  model is scored on it first, and the adapted model is written only from an epoch that beats
  it. Its latency is then recalibrated on the recordings.
- **Suggested settings.** The tempo range (5th to 95th percentile, ±5%) and the most common
  meter of the recordings go into the model's `beatref.suggest` block; the module offers to
  apply them. `beatref.adapted` records the base model and the amount of material.
- **GPU.** `train` and `finetune` use CUDA when torch finds it.

**Results**, 2026-10-28: leave one Ballroom style out. A base model is trained with the Stage 2
recipe on GTZAN and Ballroom without that style. The style's clips are split in two by a hash of
their name: the adaptation pool and the test half. The base model is then adapted with the
defaults on the first k clips of the pool (k = 10 is about 4 minutes) and scored on the test
half with automatic downbeat on. Adapted models use their suggested settings. *GTZAN F* scores
every tenth GTZAN clip with default settings, which shows how much the model forgot.

| Waltz (60 test clips) | F | CMLt | AMLt | Bar F | GTZAN F |
| --- | --- | --- | --- | --- | --- |
| Base (never heard a waltz) | 0.699 | 0.542 | 0.717 | 0.468 | 0.836 |
| Base + suggested settings | 0.707 | 0.563 | 0.723 | 0.470 | |
| Adapted, 5 clips (2 min) | 0.722 | 0.692 | 0.760 | 0.538 | 0.834 |
| **Adapted, 10 clips (4 min)** | **0.777** | **0.754** | **0.806** | 0.612 | 0.839 |
| Adapted, all 50 clips (21 min) | 0.777 | 0.743 | 0.790 | 0.622 | 0.831 |
| Shipped model (trained on every waltz, test half included) | 0.775 | 0.731 | 0.794 | 0.650 | |

| Quickstep (41 test clips) | F | CMLt | AMLt | Bar F | GTZAN F |
| --- | --- | --- | --- | --- | --- |
| Base (never heard a quickstep) | 0.643 | 0.096 | 0.774 | 0.243 | 0.841 |
| Base + suggested settings | 0.771 | 0.430 | 0.742 | 0.379 | |
| Adapted, 10 clips, default settings | 0.635 | 0.157 | 0.771 | 0.229 | |
| Adapted, 5 clips (2 min) | 0.917 | 0.799 | 0.879 | 0.507 | 0.808 |
| **Adapted, 10 clips (4 min)** | **0.909** | **0.791** | **0.875** | 0.632 | 0.811 |
| Adapted, all 41 clips (17 min) | 0.921 | 0.828 | 0.861 | 0.626 | 0.807 |
| Shipped model (trained on every quickstep) | 0.622 | 0.123 | 0.769 | 0.295 | |

The two styles fail in different ways, and learning mode fixes both. The waltz problem is in
the network: the settings change little, and four minutes of material bring it level with a
model that trained on every waltz. Quickstep sits at 200 BPM and above, outside the default
preferred range, so every model tracks it at half tempo (CMLt about 0.1), even one that trained
on it. The suggested tempo range fixes most of the level, and adaptation then fixes the phase
and the bars. Past about 4 minutes, more material adds little.

Variations on the 10-clip run (F / bar F / GTZAN F; defaults are `layers=3`, `l2sp=1e-4`):

| Variation | Waltz | Quickstep |
| --- | --- | --- |
| Defaults | 0.777 / 0.612 / 0.839 | 0.909 / 0.632 / 0.811 |
| `l2sp=0` | 0.773 / 0.605 / 0.828 | 0.913 / 0.642 / 0.792 |
| `l2sp=1e-3` | 0.758 / 0.548 / 0.840 | 0.891 / 0.553 / 0.825 |
| `layers=1` (output layer only) | 0.736 / 0.528 / 0.840 | 0.847 / 0.436 / 0.833 |
| `layers=0` (all 110k parameters) | 0.744 / 0.615 / 0.817 | 0.935 / 0.726 / 0.682 |
| `--replay` base data, `replay=0.5` | 0.752 / 0.546 / 0.842 | 0.885 / 0.560 / 0.831 |

Adapting everything fits a style best but forgets the most (GTZAN F drops by 0.16 after
Quickstep), and adapting only the output layer is too weak. A stronger pull or replay trades
some of the gain for less forgetting. An adapted model is meant for the material it was adapted
to, and the patch keeps the built-in model one click away. An earlier test that left GTZAN's
jazz out showed little to gain: the shipped model, which trained on the test clips, scores only
0.017 F above a model that never heard jazz, so the genre's errors are not ones more jazz data
fixes.

## Results

F, CMLt and AMLt come from mir_eval, with the first 5 s trimmed. Offset is the signed error of
matched beats (negative means early). Lock is the time to the first of four consecutive correct
beats. Locked is the share of beats after 5 s with the Lock gate high. Bar F is the F-measure of
the bar 1 triggers with `auto_downbeat` on.

**Neural activation**, 2026-10-26: 4-fold cross-validation, so every clip is scored by a model
that never saw it (`train --fold i/4`, then `eval --fold i/4`). The DSP rows are the same clips
with the default ODF.

| Set | Clips | Activation | F | CMLt | AMLt | Offset (ms) | Lock (s) | Locked | Bar F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ballroom | 698 | neural | **0.856** | **0.742** | **0.857** | +0.5 ± 16.1 | 3.7 | 98% | **0.526** |
| | | DSP | 0.769 | 0.574 | 0.789 | −3.2 ± 18.7 | 4.7 | 89% | 0.345 |
| [GTZAN](https://huggingface.co/datasets/marsyas/gtzan) ([annotations](https://github.com/TempoBeatDownbeat/gtzan_tempo_beat)) | 998 | neural | **0.835** | **0.739** | **0.861** | −2.5 ± 18.3 | 3.5 | 96% | **0.420** |
| | | DSP | 0.749 | 0.592 | 0.800 | −12.4 ± 19.5 | 4.8 | 86% | 0.297 |
| Synthetic | 24 | neural | 0.924 | 0.825 | 0.933 | +3.7 ± 7.8 | 5.3 | 98% | 0.617 |

Fold F ranges from 0.844 to 0.873 on Ballroom and 0.823 to 0.844 on GTZAN. The synthetic set
was never trained on; its row uses the shipped model (`plugin/res/models/beat.json`), which is
trained on all of Ballroom and GTZAN with the same recipe. GTZAN `jazz.00054` is unreadable and
left out. Most of the CMLt gain comes from the metrical level and the phase. Clips tracked at
the wrong level or off the beat (AMLt above 0.8, CMLt below 0.3) drop from 105 to 64 on
Ballroom and from 173 to 88 on GTZAN. The synthetic `sparse` grooves, where the ODF locks off
the beat, now score F 0.83 to 0.99. Bars are still the weak point: offline downbeat trackers
reach roughly 0.8 to 0.9 on Ballroom.

**DSP**, 2026-10-25:

| Set | Clips | F | CMLt | AMLt | Offset (ms) | Lock (s) | Locked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Ballroom](https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) ([annotations](https://github.com/CPJKU/BallroomAnnotations)) | 698 | 0.769 | 0.574 | 0.789 | −3.2 ± 18.7 | 4.7 | 89% |
| Synthetic (`beatref synth`, seed 1) | 24 | 0.752 | 0.640 | 0.876 | −7.2 ± 8.5 | 3.2 | 90% |

Ballroom F-measure for the other onset detection functions: spectral flux 0.726, complex
domain 0.568, HFC 0.568. With no music the lock stays off: silence is locked 0% of the time,
white noise 9%, and a sustained pad 3%.

Failure modes of the DSP activation (the network reduces the first two, see above):

- **Metrical level.** Slow material (80–95 BPM) is often tracked at double tempo, and the
  transition model cannot leave the wrong level once it's locked. The ×2/÷2 buttons cover this
  on the module. A `restart` probability exists, but it hurt scores at every value tried.
- **Off-beat lock.** This happens on very sparse, syncopated grooves, for example the
  synthetic `sparse` pattern: a kick on 1 and a snare on the "and" of 3.
- **Bars.** The low band is weak downbeat evidence; with `auto_downbeat` off, bars count from
  the first beat. The module has Bar Reset.

Tap and Clock events are quantised to frames here (about 5 ms of bias). The module will apply
them at audio rate.
