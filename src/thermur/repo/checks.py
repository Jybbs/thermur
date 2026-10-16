"""
Holds `Check`, the class every check `repo:audit` runs extends, and the
checks themselves, each one invariant the audit holds the checkout to and
each found through `Check.__subclasses__()` rather than through a list kept
beside them.
"""

from collections            import Counter
from collections.abc        import Iterator
from filecmp                import cmp
from itertools              import product
from packaging.requirements import Requirement
from packaging.utils        import canonicalize_name
from pathlib                import Path
from pydantic               import BaseModel, ValidationError
from re     import findall, finditer, search
from typing import ClassVar
from yaml   import NodeEvent, YAMLError, parse

from thermur.docs.outputs  import Outputs
from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Finding, Parity


class Check(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One invariant the audit holds the checkout to, read through `checkout`.
    """

    checkout: Checkout
    """
    The clone the check reads.
    """

    @property
    def findings(self) -> list[Finding]:
        """
        Collects every finding `scan` makes, reporting a document that fails
        to validate as one finding per field on the file the error's title
        names, and letting through an error whose title names no file.
        """
        try:
            return list(self.scan())
        except ValidationError as error:
            if not (self.checkout.root / error.title).is_file():
                raise

            return [
                Finding(
                    file    = Path(error.title),
                    message = (
                        f"`{'.'.join(map(str, detail['loc']))}` in "
                        if detail["loc"]
                        else ""
                    ) + f"`{error.title}`: {detail['msg']}"
                )
                for detail in error.errors()
            ]

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each declaration a file restates with the one it restates,
        which a check comparing no restated declarations leaves empty.
        """
        return []

    def scan(self) -> Iterator[Finding]:
        """
        Reads the checkout for every place it breaks this invariant, which
        for a check comparing restated declarations is each pair whose copy
        reads anything but the original.
        """
        yield from filter(None, (pair.finding for pair in self.pairs))


class AnchorCheck(Check):
    """
    No composite action's manifest carries a YAML anchor, since GitHub's
    action-manifest parser rejects one before any step of a job loading the
    action runs, whereas a workflow may carry one.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each anchor a manifest declares or reads through an alias,
        once and in the order PyYAML's parse events carry them, and where
        PyYAML cannot parse the manifest, names the error it reports,
        keeping each line of its message but the indented ones that locate
        it in the text.
        """
        for path in self.checkout.actions:
            try:
                events = list(parse(self.checkout.read(path)))
            except YAMLError as error:
                yield Finding(
                    file    = path,
                    message = f"PyYAML cannot parse `{path}`: "
                    + ", ".join(
                        line
                        for line in str(error).splitlines()
                        if not line.startswith(" ")
                    )
                )
                continue

            for name in dict.fromkeys(
                event.anchor
                for event in events
                if isinstance(event, NodeEvent) and event.anchor
            ):
                yield Finding(
                    file    = path,
                    message = f"`{path}` carries the YAML anchor `&{name}`, which "
                    "GitHub's action-manifest parser rejects"
                )


class BinCheck(Check):
    """
    Every program in the folder the `_.path` of `.mise/config.toml` puts on
    the path is a regular file holding the same bytes as the first of them.

    Git checks a symlink out as a plain, non-executable file holding its
    target's name wherever `core.symlinks` is off, so a shell looking the
    program up on the path skips it and runs any program of that name the
    machine already holds.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each program that is a symlink, then each copy whose bytes
        differ from the first copy's.
        """
        root   = self.checkout.root
        linked = [path for path in self.checkout.scripts if (root / path).is_symlink()]
        for path in linked:
            yield Finding(
                file    = path,
                message = f"`{path}` is a symlink rather than a copy"
            )

        copies = [path for path in self.checkout.scripts if path not in linked]
        for path in copies[1:]:
            if not cmp(root / copies[0], root / path, shallow=False):
                yield Finding(
                    file    = path,
                    message = f"`{path}` differs byte for byte from `{copies[0]}`"
                )


class CommitCheck(Check):
    """
    Every action from another repository is pinned to one commit across the
    workflows and the composite actions naming it.
    """

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each step pinning an action with the first step pinning the
        same action, in path order.
        """
        pinned = [(path, step.pin) for path, step in self.checkout.steps if step.pin]
        first  = {}
        for path, (action, commit) in pinned:
            first.setdefault(action, (path, commit))

        return [
            Parity(
                copied   = commit,
                file     = path,
                label    = f"The pin on `{action}` in `{path}`",
                origin   = f"the pin on it in `{first[action][0]}`",
                original = first[action][1]
            )
            for path, (action, commit) in pinned
        ]


class GateCheck(Check):
    """
    Every workflow ends on a job under the key `gate`, which names the check
    run it reports, runs under `always()`, and waits for every other job, so
    the one check a ruleset requires carries every job's verdict.

    A gate job left out of a failed run is reported as skipped, which a
    ruleset counts as passing, so the gate runs whatever the other jobs
    end in.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each workflow ending on no gate job, then each gate job
        that sets no name, runs under another condition, or leaves a job
        unwaited.
        """
        for path, workflow in self.checkout.workflows.items():
            if not (gate := workflow.gate):
                yield Finding(file=path, message=f"`{path}` ends on no `gate` job")
                continue

            if not gate.name:
                yield Finding(
                    file    = path,
                    message = f"The `gate` job in `{path}` sets no name"
                )

            if not gate.always:
                yield Finding(
                    file    = path,
                    message = f"The `gate` job in `{path}` runs under "
                    f"`{gate.condition}` rather than `always()`"
                )

            for key in sorted(workflow.gated.keys() - gate.needs):
                yield Finding(
                    file    = path,
                    message = f"The `gate` job in `{path}` does not wait for `{key}`"
                )


class GuideCheck(Check):
    """
    The contributor guide's checks table lists each row of every workflow
    running on a pull request under the name its check run takes, beside the
    command running the row's task, and lists no other check.
    """

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs the command the guide gives each row's check with the command
        running the task the row runs.
        """
        guide    = self.checkout.guide
        commands = {task.name: task.command for task in self.checkout.tasks}
        return [
            Parity(
                copied   = guide.checks.get(name),
                file     = guide.file,
                label    = f"The command beside `{name}` in `{guide.file}`",
                origin   = f"the command running `{task}`",
                original = commands.get(task, f"mise run {task}")
            )
            for name, task in self.checkout.checks
        ]

    def scan(self) -> Iterator[Finding]:
        """
        Names each check the guide lists that no row of a pull-request
        workflow runs, then each row whose command the guide gives
        otherwise.
        """
        guide = self.checkout.guide
        run   = {name for name, _ in self.checkout.checks}
        for name in guide.checks:
            if name not in run:
                yield Finding(
                    file    = guide.file,
                    message = f"`{guide.file}` lists `{name}`, a check no row of a "
                    "pull-request workflow runs"
                )

        yield from super().scan()


class ImageCheck(Check):
    """
    Every job runs on one runner image, so a reviewed bump moves the tools
    preinstalled under every job together.
    """

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each job with the first job of the first workflow, in path
        order.
        """
        jobs = self.checkout.jobs
        return [
            Parity(
                copied   = job.image,
                file     = path,
                label    = f"The runner image of the `{key}` job in `{path}`",
                origin   = f"the one the `{opening}` job in `{origin}` runs on",
                original = first.image
            )
            for origin, opening, first in jobs[:1]
            for path, key, job in jobs
        ]


class KitCheck(Check):
    """
    The tool subsets the jobs install through a composite action are the
    subsets a job saves the archive of, since the mise action keys each
    archive by its subset, so a subset no job saves is downloaded afresh on
    every workflow run, and a saved subset no job installs is never read.
    """

    @property
    def subsets(self) -> tuple[dict[str, Path], dict[str, Path]]:
        """
        Maps each subset the jobs install through a composite action without
        saving it, and then each subset a job saves, to the first workflow
        naming it, leaving out a step whose `tools` names no subset, for
        which the composite installs nothing.
        """
        installed, saved = {}, {}
        for path, _, job in self.checkout.jobs:
            for step in job.steps:
                if step.uses in self.checkout.composites:
                    kept = saved if step.saves else installed
                    for subset in filter(
                        None,
                        job.expand(step.inputs.get("tools", ""))
                    ):
                        kept.setdefault(subset, path)

        return installed, saved

    def scan(self) -> Iterator[Finding]:
        """
        Names each subset the jobs install that no job saves, on the
        workflow installing it, then each subset a job saves that no job
        installs, on the workflow saving it.
        """
        installed, saved = self.subsets
        for subset, path in sorted(installed.items()):
            if subset not in saved:
                yield Finding(
                    file    = path,
                    message = f"`{path}` installs the `{subset}` subset, whose archive "
                    "no job saves"
                )

        for subset, path in sorted(saved.items()):
            if subset not in installed:
                yield Finding(
                    file    = path,
                    message = f"`{path}` saves the archive of the `{subset}` subset, "
                    "which no job installs"
                )


class LabelCheck(Check):
    """
    Every file naming a label agrees with `.github/labels.toml`, meaning the
    release-notes categories file each label it declares under one category,
    the issue templates and the contributor guide name only labels it
    declares, and the guide describes each label as the registry does.
    """

    @property
    def named(self) -> dict[Path, list[str]]:
        """
        Maps each file naming labels to every label it names, in the order
        it names them, meaning the release-notes categories, each issue
        template, and the contributor guide.
        """
        guide, release = self.checkout.guide, self.checkout.release
        return (
            {release.file: release.names}
            | {
                path: template.labels
                for path, template in self.checkout.templates.items()
            }
            | {guide.file: list(guide.labels)}
        )

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs the description the contributor guide gives each label with
        the one the registry declares.
        """
        guide, labels = self.checkout.guide, self.checkout.labels
        return [
            Parity(
                copied   = guide.labels.get(label.name),
                file     = guide.file,
                label    = f"The description of `{label.name}` in `{guide.file}`",
                origin   = f"its description in `{labels.file}`",
                original = label.description
            )
            for label in labels.rows
        ]

    def scan(self) -> Iterator[Finding]:
        """
        Names each label a file names that the registry does not declare,
        then each declared label the categories file under any number of
        categories but one, then each label the guide describes otherwise.
        """
        labels = self.checkout.labels
        for file, names in self.named.items():
            for name in dict.fromkeys(names):
                if name not in labels.names:
                    yield Finding(
                        file    = file,
                        message = f"`{file}` names `{name}`, a label `{labels.file}` "
                        "does not declare"
                    )

        release = self.checkout.release
        filed   = Counter(release.names)
        for name in labels.names:
            if filed[name] != 1:
                yield Finding(
                    file    = release.file,
                    message = f"`{release.file}` files `{name}` under "
                    f"{filed[name] or 'no'} release-notes categories rather than one"
                )

        yield from super().scan()


class LockfileCheck(Check):
    """
    Every package manager a task or a program on the path runs installs
    from its lockfile, meaning each `uv run` opens its arguments on
    `--exact --locked`, so it installs exactly what `uv.lock` resolves and
    stops where the lockfile lags its manifest, each `bun install` carries
    `--frozen-lockfile` or `--lockfile-only`, so it installs exactly what
    `docs/bun.lock` resolves or rewrites the lockfile alone, and none runs
    `bunx`, which can run a copy of a tool from outside `docs/node_modules`.
    """

    rules: ClassVar[dict[str, str]] = {
        r"(?m)^(?!.*(?:--frozen-lockfile|--lockfile-only)).*\bbun[ \t]+install\b":
            "`bun install` with neither `--frozen-lockfile` nor `--lockfile-only`",

        r"\bbunx\b":
            "`bunx`, which can run a copy of a tool from outside `docs/node_modules`",

        r"\buv[ \t]+run\b(?![ \t]+--exact[ \t]+--locked\b)":
            "`uv run` without `--exact --locked` opening its arguments"
    }
    """
    Each pattern no file's commands may match, beside what a match runs.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each file whose commands match a pattern of `rules`, once for
        each pattern they match.
        """
        for (path, lines), (pattern, runs) in product(
            self.checkout.commands,
            self.rules.items()
        ):
            if search(pattern, lines):
                yield Finding(file=path, message=f"`{path}` runs {runs}")


class ParityCheck(Check):
    """
    Every file restating the Python version, the uv release, the license,
    or the release `CITATION.cff` cites reads the value `pyproject.toml`
    or `.mise/config.toml` declares, and every pin of the site's packages
    restating another pin reads that pin's version.

    The manifest's `requires-python` floor is the Python version the
    README's badge, the formatter's `target-version`, and the minor of
    the `python` mise pins each restate. The `uv` mise pins is the release
    `[tool.uv]` requires, and the manifest's license and authors are what
    the README's badge and the title and the copyright line of `LICENSE`
    restate. `CITATION.cff` restates the manifest's version, license,
    repository, and authors. Vitest's coverage provider restates the release
    of Vitest it ships beside, and the major of Node's types restates the
    major of the `node` mise pins.
    """

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each restated declaration with the one it restates.
        """
        citation     = self.checkout.citation
        config       = self.checkout.config
        manifest     = self.checkout.manifest
        package      = self.checkout.package
        license_path = Path("LICENSE")
        readme       = self.checkout.read(manifest.readme)
        notice       = self.checkout.read(license_path)
        badge        = self.extract(r"badge/License-((?:--|[^-])+)-", readme)
        python       = f"the floor of `requires-python` in `{manifest.file}`"

        return [
            Parity(
                copied   = self.extract(r"badge/python-(\d+\.\d+)\+", readme),
                file     = manifest.readme,
                label    = f"The Python badge in `{manifest.readme}`",
                origin   = python,
                original = manifest.floor
            ),
            Parity(
                copied   = manifest.target_version,
                file     = manifest.file,
                label    = "The `target-version` in `[tool.prose]`",
                origin   = python,
                original = manifest.floor
            ),
            Parity(
                copied   = config.python,
                file     = config.file,
                label    = f"The minor of the `python` pin in `{config.file}`",
                origin   = python,
                original = manifest.floor
            ),
            Parity(
                copied   = manifest.required_version,
                file     = manifest.file,
                label    = "The `required-version` in `[tool.uv]`",
                origin   = f"the `uv` pin in `{config.file}` as a `==` requirement",
                original = f"=={config.uv}"
            ),
            Parity(
                copied   = badge.replace("--", "-") if badge else None,
                file     = manifest.readme,
                label    = f"The License badge in `{manifest.readme}`",
                origin   = f"the `license` in `{manifest.file}`",
                original = manifest.license
            ),
            Parity(
                copied   = notice.partition("\n")[0],
                file     = license_path,
                label    = f"The title of `{license_path}`",
                origin   = f"the full name of the `license` in `{manifest.file}`",
                original = f"{manifest.license} License"
            ),
            Parity(
                copied   = self.extract(r"Copyright \(c\) [\d-]+ (.+)", notice),
                file     = license_path,
                label    = f"The copyright holder list in `{license_path}`",
                origin   = f"the author list in `{manifest.file}`",
                original = manifest.holders
            ),
            *(
                Parity(
                    copied   = copied,
                    file     = citation.file,
                    label    = f"The `{key}` in `{citation.file}`",
                    origin   = f"the `{name}` in `{manifest.file}`",
                    original = original
                )
                for key, copied, name, original in (
                    ("license", citation.license, "license", manifest.license),
                    (
                        "repository-code", citation.repository,
                        "Repository", manifest.repository
                    ),
                    ("version", citation.version, "version", manifest.version)
                )
            ),
            Parity(
                copied   = citation.holders or None,
                file     = citation.file,
                label    = f"The author list in `{citation.file}`",
                origin   = f"the author list in `{manifest.file}`",
                original = manifest.holders
            ),
            Parity(
                copied   = package.coverage,
                file     = package.file,
                label    = f"The pin on `@vitest/coverage-v8` in `{package.file}`",
                origin   = "the pin on `vitest` there",
                original = package.vitest
            ),
            Parity(
                copied   = package.types,
                file     = package.file,
                label    = f"The major of the pin on `@types/node` in `{package.file}`",
                origin   = f"the major of the `node` pin in `{config.file}`",
                original = config.node
            )
        ]

    @staticmethod
    def extract(pattern: str, text: str) -> str | None:
        """
        Reads the first group `pattern` captures in `text`, or returns
        `None` where nothing matches.
        """
        return match[1] if (match := search(pattern, text)) else None


class PinCheck(Check):
    """
    The hatchling build requirement and every build constraint `[tool.uv]`
    declares each name one version, since uv resolves hatchling and its own
    dependencies outside `uv.lock`.
    """

    @staticmethod
    def exact(requirement: Requirement) -> bool:
        """
        Tells whether `requirement` names one version, meaning a single `==`
        specifier carrying no `.*` wildcard.
        """
        return [
            (specifier.operator, specifier.version.endswith(".*"))
            for specifier in requirement.specifier
        ] == [("==", False)]

    def scan(self) -> Iterator[Finding]:
        """
        Names a `[build-system]` holding no exact pin on hatchling, under
        any spelling its normalized name allows, then each build constraint
        naming a range.
        """
        manifest = self.checkout.manifest
        if not any(
            canonicalize_name(requirement.name) == "hatchling"
            and self.exact(requirement)
            for requirement in manifest.requires
        ):
            yield Finding(
                file    = manifest.file,
                message = "The `requires` of `[build-system]` holds no exact pin on "
                "hatchling"
            )

        for constraint in manifest.constraints:
            if not self.exact(constraint):
                yield Finding(
                    file    = manifest.file,
                    message = f"The build constraint `{constraint}` in `[tool.uv]` "
                    "is not an exact pin"
                )


class ProvisionCheck(Check):
    """
    No step passes a composite action an input its manifest leaves out,
    which GitHub only warns about before running the action on its defaults,
    and a push changing any file the action hashes into an archive's key
    runs each workflow saving that archive, so the change saves it afresh.

    The `cache_key` a composite action passes `jdx/mise-action` ends on
    a `-` and a `hashFiles` call, holding before them only the template
    variables that stay fixed across the saves of one tool subset, since
    `gha:prune` keeps the newest of the entries one branch saved whose keys
    match up to that `-`.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each step passing an undeclared input, then each hashed file
        whose change runs no workflow saving the archive, then each cache
        key `gha:prune` cannot group by tool subset.
        """
        for name, path in self.checkout.composites.items():
            action = self.checkout.actions[path]
            for home, step in self.checkout.steps:
                if step.uses == name:
                    for key in sorted(step.inputs.keys() - action.inputs.keys()):
                        yield Finding(
                            file    = home,
                            message = f"`{home}` passes `{name}` the input `{key}`, "
                            f"which `{path}` does not declare"
                        )

            calls = findall(r"hashFiles\(([^)]*)\)", self.checkout.read(path))
            for (home, workflow), file in product(
                self.checkout.workflows.items(),
                findall(r"'([^']+)'", " ".join(calls))
            ):
                if workflow.saves(name) and not workflow.pushes(file):
                    yield Finding(
                        file    = home,
                        message = f"The push filter of `{home}` leaves out `{file}`, "
                        f"which `{path}` hashes"
                    )

            for step in action.steps:
                if not step.uses.startswith("jdx/mise-action@"):
                    continue

                if (archive := step.archive) is None:
                    yield Finding(
                        file    = path,
                        message = f"`{path}` passes `jdx/mise-action` no `cache_key` "
                        "ending on a `-` and a `hashFiles` call, whose hash "
                        "`gha:prune` drops to group the saves of one tool subset"
                    )
                    continue

                for variable in finditer(r"\$?\{\{\s*(.*?)\s*\}\}", archive):
                    if variable[1] not in {
                        "bootstrap_hash", "cache_key_prefix", "install_args_hash",
                        "mise_env", "platform"
                    }:
                        yield Finding(
                            file    = path,
                            message = f"The `cache_key` in `{path}` holds "
                            f"`{variable[0]}` before its `hashFiles` call, whereas "
                            "only a value fixed across the saves of one tool subset "
                            "keeps `gha:prune` grouping them"
                        )


class RenderCheck(Check):
    """
    Every file the render task writes into `docs/` reads what the
    declaration in the package it is written from would write now, so the
    site, which calls no Python, never reads a stale copy.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each written file whose text on disk has drifted from its
        declaration, a missing file among them.
        """
        for path in Outputs(root=self.checkout.root).drifted:
            yield Finding(
                file    = path,
                message = f"`{path}` has drifted from the declaration it is written "
                "from, which `mise render` writes again"
            )


class RulesetCheck(Check):
    """
    Every check a ruleset under `.github/rulesets/` requires is the name of
    the gate a workflow running on a pull request ends on.

    A required check no pull request reports leaves every pull request
    waiting on it, so this check fails the pull request that renames a gate
    before the rename strands the ruleset.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each check a ruleset requires that no gate of a workflow
        running on a pull request names.
        """
        gates = {
            workflow.gate.name
            for workflow in self.checkout.workflows.values()
            if workflow.on_pull_request and workflow.gate
        }
        for path, ruleset in self.checkout.rulesets.items():
            for context in sorted(ruleset.contexts - gates):
                yield Finding(
                    file    = path,
                    message = f"`{path}` requires the `{context}` check, which no "
                    "gate of a workflow running on a pull request names"
                )


class ScriptCheck(Check):
    """
    Every package a task script's inline metadata declares that `uv.lock`
    also resolves is pinned to the version `uv.lock` resolves, so the script
    and the suite run one release of it.
    """

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each such requirement with the exact pin on the version
        `uv.lock` resolves.
        """
        root     = self.checkout.root
        versions = self.checkout.lock.versions
        return [
            Parity(
                copied   = str(requirement.specifier) or None,
                file     = file,
                label    = f"The pin on `{name}` in `{file}`",
                origin   = "the version `uv.lock` resolves",
                original = f"=={versions[name]}"
            )
            for task in self.checkout.tasks
            if (file := task.file and task.file.relative_to(root))
            for requirement in task.dependencies
            if (name := canonicalize_name(requirement.name)) in versions
        ]


class SummaryCheck(Check):
    """
    Only a gate job writes to the workflow step summary, and no step inside
    a composite action does, so a workflow run's summary page comes from
    one renderer.

    A step writes to it by naming the variable that holds its path or by
    running `gha:brief`, which reads that variable, so both are matched.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each job other than a gate, and each composite action, holding
        a step that writes to the summary.
        """
        writers = [
            (path, f"The `{key}` job in `{path}`", job.steps)
            for path, workflow in self.checkout.workflows.items()
            for key, job in workflow.gated.items()
        ] + [
            (path, f"The composite action `{path}`", action.steps)
            for path, action in self.checkout.actions.items()
        ]
        for path, subject, steps in writers:
            if any(step.summarizes for step in steps):
                yield Finding(
                    file    = path,
                    message = f"{subject} writes to the workflow step summary, which "
                    "only a `gate` job writes"
                )


class TaskCheck(Check):
    """
    Every task the checkout declares passes `mise tasks validate`, which
    reports an alias two tasks share and a dependency naming no task, among
    other problems, each as an error or as a warning.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each problem mise reports on a task the checkout declares,
        at the level mise reports it, on the file holding or declaring the
        task.
        """
        files = {
            task.name: task.path.relative_to(self.checkout.root)
            for task in self.checkout.tasks
        }
        for issue in self.checkout.issues:
            if issue.task in files:
                yield issue.finding(files[issue.task])


class WheelCheck(Check):
    """
    Every `packages` entry of the wheel holds a module, and every script
    `[project.scripts]` declares targets a module the wheel carries, since
    hatchling skips a `packages` entry naming nothing and still writes the
    wheel.
    """

    @property
    def modules(self) -> dict[Path, set[str]]:
        """
        Maps each `packages` entry of the wheel to the dotted name of every
        module under it, the import name opening on the entry's last folder
        as hatchling installs it.
        """
        root = self.checkout.root
        return {
            entry: {
                ".".join(
                    path.relative_to((root / entry).parent).with_suffix("").parts
                ).removesuffix(".__init__")
                for path in (root / entry).rglob("*.py")
            }
            for entry in self.checkout.manifest.packages
        }

    def scan(self) -> Iterator[Finding]:
        """
        Names each `packages` entry holding no module, then each script
        targeting a module no entry holds.
        """
        manifest = self.checkout.manifest
        modules  = self.modules
        for entry, names in modules.items():
            if not names:
                yield Finding(
                    file    = manifest.file,
                    message = f"The wheel `packages` entry `{entry}` holds no module"
                )

        carried = set().union(*modules.values())
        for script, target in manifest.scripts.items():
            if (module := target.partition(":")[0]) not in carried:
                yield Finding(
                    file    = manifest.file,
                    message = f"The `{script}` script targets `{module}`, which the "
                    "wheel does not carry"
                )
