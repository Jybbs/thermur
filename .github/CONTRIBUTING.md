# Contributing to Thermur

Every change to *Thermur* starts from an issue, lands on a branch named for that issue, and reaches `main` through a pull request that Jybbs reviews and squash-merges. This guide follows that path from a fresh clone to a merged pull request and closes on the labels that sort each issue and pull request.

## Setting Up the Clone

1. Build the locked environment through the commands under [Quick Start](../README.md#-quick-start) in the README, which install the Python and uv `.mise/config.toml` pins and then the packages `uv.lock` resolves.
2. Run `mise doctor project`, which names any condition the clone still lacks beside the command that supplies it.

## From an Issue to a Pull Request

1. Open an issue through the [spec or bug template](https://github.com/Jybbs/thermur/issues/new/choose), naming what is wrong and what has to change, which Jybbs labels from the table under Labels below.
2. Branch from `main` as `<issue>/<slug>`, the issue's number followed by at most three words from its title (*`12/seeded-trajectories`*).
3. Write the change beside the tests that pin it, then run `mise format` and run `mise check`, `mise test`, and `mise audit` until each passes. After editing `pyproject.toml` or `.mise/config.toml`, run `mise relock`, so the lockfile it rewrites lands in the same commit.
4. Commit in the form `type(scope): description`, its type one of `feat`, `fix`, `refactor`, `chore`, `docs`, or `test`, and its scope the part of the package the commit touches (*`cli`, `config`, `controller`, `environment`, `metrics`*) or `docs`, `build`, or `ci`.
5. Open a pull request titled `[<issue>] <the issue's title>`, filling in the sections its template carries, whose Related Issues line closes the issue on merge.

Jybbs squash-merges each pull request into one commit on `main` titled after the pull request, which is why the title rather than any commit on the branch carries the issue's number.

## Labels

Each issue and pull request takes the label of every domain or shared concern its work touches, beside `🐞 bug` where the work repairs a defect and `🔬 discovery` where it ends in a finding. GitHub sorts each merged pull request into a release's notes under the first of its labels in the order `.github/release.yml` gives, and `.github/labels.toml` declares every label below:

| **Label** | **Covers** |
|---|---|
| `🐞 bug` | Wrong output or a broken run in any domain |
| `🦜 cli` | The command line |
| `🧱 config` | The settings model and the Hamilton graph every domain reads |
| `🪶 controller` | The drones' murmuration controller and its response to fire |
| `🔬 discovery` | Work that ends in a finding rather than a change to the code |
| `📚 docs` | The README and the docs site |
| `🌪️ environment` | The WRF-SFIRE fields and the physics that moves the flock |
| `🌡️ metrics` | The measured properties of the flock and their calibration |
| `🧰 tooling` | The packaging, dependencies, workflows, and release |
