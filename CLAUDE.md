# Working on this repo

- `plugin/` is the C++ Rack module, `python/` the reference tracker and evaluation harness. After any
  engine change, run `make -C plugin beattrack` and `beatref compare` (see `plugin/README.md`).
- **Every PR with non-cosmetic code changes bumps `version` in `plugin/plugin.json`** (semver; the
  major stays 2 to match Rack) and updates the roadmap/status in `README.md` and any other docs the
  change affects. Cosmetic means comments, formatting or docs only.
- Merging a version that has no `v<version>` tag to `main` makes CI tag it and publish a GitHub
  Release with the lin-x64, win-x64, mac-x64 and mac-arm64 `.vcvplugin` builds
  (`.github/workflows/ci.yml`). Nothing else is needed to release.
