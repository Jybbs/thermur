# Contributing to Thermur

Every change to *Thermur* starts from an issue, lands on a branch named for that issue, and reaches `main` through a pull request that is reviewed and then squash-merged. This guide follows that path from a fresh clone to a merged pull request, then sets out the style the code takes, how a page cites a study, the checks each pull request runs, and the labels that sort each issue and pull request.

## Setting Up the Clone

1. Build the locked environment through the commands under [Quick Start](../README.md#-quick-start) in the README, which install every tool `.mise/config.toml` pins and then the packages `uv.lock` resolves.
2. Run `mise doctor project`, which names any condition the clone still lacks beside the command that supplies it.

## From an Issue to a Pull Request

1. Open an issue through the [spec or bug template](https://github.com/Jybbs/thermur/issues/new/choose), naming what is wrong and what has to change, and the issue takes its labels from the table under Labels below.
2. Branch from `main` as `<issue>/<slug>`, the issue's number followed by at most three words from its title (*`12/seeded-trajectories`*).
3. Write the change beside the tests that pin it, then run `mise format` and run `mise ci`, which runs every check under [Checks](#checks) below, until it passes. After editing `pyproject.toml`, `.mise/config.toml`, or `docs/package.json`, run `mise relock`, so the lockfile it rewrites lands in the same commit.
4. Commit in the form `type(scope): description`, its type one of `feat`, `fix`, `refactor`, `chore`, `docs`, or `test`, and its scope the part of the package the commit touches (*`cli`, `config`, `controller`, `environment`, `metrics`*) or `docs`, `build`, or `ci`.
5. Open a pull request titled `[<issue>] <the issue's title>`, filling in the sections its template carries, whose Related Issues line closes the issue on merge.

Each pull request is squash-merged into one commit on `main` titled after the pull request, which is why the title rather than any commit on the branch carries the issue's number.

## Code Style

[*Prose*](https://prose.fyi) settles the style of the Python under `.mise/tasks`, `src`, and `tests` from the one `[tool.prose]` table in `pyproject.toml`. `mise format` rewrites the source to that style, and `mise check` reports every rewrite the formatter would make beside every lint finding. A lint finding is fixed in the source or held out with `# prose: ignore[<rule>]` on the line it anchors on, which silences that one rule on that one line. A `# prose: skip` trailing the opening line of a construct keeps the line breaks the formatter would reflow there and the construct's place among its siblings, while the lines inside it still take the formatter's other rewrites. The marker silences no lint finding.

## Citing a Study

A page cites published work through a footnote whose label is the study's key in `docs/references.bib`, the first author's surname and the year in lowercase (*`cavagna2010`*), with a letter after the year telling apart two studies one author published that year (*`ballerini2008a`, `ballerini2008b`*). The file holds one entry per study, in the order of its key, each carrying an `annote` on what the study measured, how close it sits to *Thermur*, and how much it can shape the composite, and a `priority` of `prio1`, `prio2`, or `prio3`, the values JabRef writes in its priority field for high, medium, and low. Each study is read in full before it enters, whereas one that no open copy reaches enters from its abstract, with an `annote` that says so and claims nothing about the study the abstract does not state. Wherever a journal has published a study, the entry records that version rather than a preprint, since peer review has checked it and a preprint's figures can differ from the published ones. A preprint enters only where no journal has published the study.

1. Add the study to `docs/references.bib` as an `@article`, or, for a preprint, as the `@misc` arXiv's BibTeX export writes, naming its `archiveprefix`, `eprint`, and `primaryclass`, beside the `doi` arXiv assigns it.
2. Type only the marker in the page (*`[^cavagna2010]`*), then run `mise render`, which writes the definition of every study the page cites at its end, in the order the page first cites each, and removes the definition of one it no longer cites.
3. Paste what `mise cite <key>` prints beneath the text of an issue, a pull request, or a comment citing the study, since no render reaches that text.

`mise audit` fails where a page's definitions differ from what `mise render` writes, and where a marker under a study's key names no entry in the file.

## Checks

Every pull request runs `🐦‍⬛ CI` except one touching only the site's own code under `docs/.vitepress/`, and one touching what the site reads runs `🐦‍⬛ Site`, whose rows each run one task, so the command beside each check below runs the same task on a laptop:

| **Check** | **Runs Locally** | **Fails When** |
|---|---|---|
| `🪵 Lockfiles` | `mise lockfile` | `uv.lock`, `.mise/mise.lock`, `docs/bun.lock`, or the lockfile beside a task script drifts from its manifest, or `.mise/mise.lock` lacks a pinned tool for the running platform |
| `⛈️ Advisories` | `mise scan` | A package `uv.lock`, `docs/bun.lock`, or a task script's lockfile resolves carries a published advisory, or the advisory service cannot be reached |
| `🪻 Prose` | `mise check` | *Prose* reports a rewrite or a lint finding, each shown as an annotation on the line it concerns |
| `☂️ Coverage` | `mise coverage` | A test fails under coverage, or the suite fails to collect |
| `🧯 Audit` | `mise audit` | A check under `src/thermur/repo/` makes a finding at the error level, such as a file restating a value otherwise than its source, or this guide naming a check or a label otherwise than the workflows or `.github/labels.toml` |
| `🗝️ Workflows` | `mise vet` | [zizmor](https://docs.zizmor.sh/) cannot parse a workflow or the composite action, or finds a hazard in one, such as template injection, an unpinned action, an over-broad permission, checkout credentials left on disk, or a cache a privileged job restores |
| `🪁 Press` | `mise press` | VitePress fails to build a page, or a link between pages leads nowhere |
| `📻 Links` | `mise links` | A link, an asset, or a heading fragment in the built site, the README, or this guide names nothing it can reach |
| `🚿 Lint` | `mise lint` | Oxlint reports a finding in the site's TypeScript, each shown as an annotation on the line it concerns |
| `🚩 Types` | `mise typecheck` | `tsc` reports a type error in the site's TypeScript, or a single-file component appears that `tsc` cannot read |
| `🪓 Cruft` | `mise cruft` | knip reports an unused file, export, or dependency in the site, or an import cycle |
| `🦺 Coverage` | `mise run site:coverage` | A site test fails under coverage, or the site's coverage falls below 95% on statements, branches, functions, or lines |

Each workflow ends on a `📡 Readout` gate that waits on every row and fails where any of them fails, so it is the one check a pull request has to pass, and `mise ci` runs every row's task once `mise doctor project` passes.

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
