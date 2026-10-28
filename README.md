# vcvtempodetector

A VCV Rack module that listens to any musical audio and outputs a steady, predictive beat clock:
beat and bar triggers, 0–10 V phase saws, BPM as CV, a multiplied/divided clock, and lock and
confidence outputs. Tracking is causal, so once locked the trigger lands on the beat, and phase
and time offsets cancel any delay downstream.

| Stage | What | Status |
| --- | --- | --- |
| 0 | Python reference tracker and evaluation harness ([`python/`](python/)) | done |
| 1 | C++ DSP module ([`plugin/`](plugin/)): onset detection, online HMM tracker, flywheel oscillator, display | done |
| 2 | Neural activations ([`python/`](python/#neural-activation-stage-2)): a causal TCN with our own weights, beat and downbeat, run through RTNeural on a worker thread; user-loadable models | done |
| 3 | Learning mode ([`python/`](python/#learning-mode-stage-3)): adapt the model to your material from audio plus a beat grid. `beatref finetune` and a Colab notebook are done (4 minutes of an unheard style brings a model level with one trained on all of it); recording from the Clock input and a model library in the module are next, then fine-tuning inside Rack | in progress |

Every stage keeps the same tracker and oscillator back end and changes only what feeds it.

## Releases

Download the `.vcvplugin` for your platform from
[Releases](https://github.com/un1tz3r0/vcvtempodetector/releases) and drop it in Rack's `plugins`
folder. CI builds every Rack platform (Linux x64, Windows x64, macOS x64 and arm64) on each PR and
push; merging a PR that bumps `version` in [`plugin/plugin.json`](plugin/plugin.json) tags
`v<version>` and publishes the release. Every non-cosmetic change bumps the version (see
[CLAUDE.md](CLAUDE.md)).

Licensed under GPL-3.0 (see [LICENSE](LICENSE)). Algorithms ported from madmom are credited in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
