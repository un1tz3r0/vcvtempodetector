# vcvtempodetector

A VCV Rack module that listens to any musical audio and outputs a steady, predictive beat clock:
beat and bar triggers, 0–10 V phase saws, BPM as CV, a multiplied/divided clock, and lock and
confidence outputs. Tracking is causal, so once locked the trigger lands on the beat, and phase
and time offsets cancel any delay downstream.

| Stage | What | Status |
| --- | --- | --- |
| 0 | Python reference tracker and evaluation harness ([`python/`](python/)) | done |
| 1 | C++ DSP module ([`plugin/`](plugin/)): onset detection, online HMM tracker, flywheel oscillator, display | done |
| 2 | Neural activations (causal CRNN/TCN, our own weights, RTNeural) | next |
| 3 | Learning mode: adapt the model to your material from audio plus a clock | planned |

Every stage keeps the same tracker and oscillator back end and changes only what feeds it.

Licensed under GPL-3.0 (see [LICENSE](LICENSE)). Algorithms ported from madmom are credited in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
