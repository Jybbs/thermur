"""
Holds `Lift`, the pins lagging their newest releases that `lock:lift`
reports, tries in a scratch clone, and writes, and `Manager`, the class each
source of pins extends, meaning each package manager, GitHub for the actions
and the runner image, and Bun for wrangler. Each manager reads its lagging
pins through the command its source ships, and the lift finds each through
`Manager.__subclasses__()` rather than through a list kept beside them.
"""

from collections.abc import Collection
from datetime        import datetime
from json            import dumps, loads
from pathlib         import Path
from pydantic        import BaseModel, Field
from re              import escape, findall, fullmatch
from subprocess      import CalledProcessError
from tempfile        import TemporaryDirectory
from typing          import ClassVar

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import (
    Audit,
    Edit,
    Lag,
    Task,
    Tree,
    read_minor,
    read_pin,
    read_release
)


def read_lags(fields: dict) -> list[Lag]:
    """
    Gathers the pins each manager reports lagging in the checkout the
    validated fields hold under `checkout`, in the order the managers are
    declared.
    """
    return [
        *manager(checkout=fields["checkout"]).lags
        for manager in Manager.__subclasses__()
    ]


class Lift(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    Every pin the checkout declares that lags its newest release, which the
    lift writes short of each pin `holds` names.
    """

    checkout: Checkout
    """
    The clone whose pins the lift reads.
    """

    holds: tuple[str, ...] = ()
    """
    The pins the lift leaves where they stand, each named as the table names
    it.
    """

    lags: list[Lag] = Field(default_factory=read_lags)
    """
    Every pin lagging its newest release, those of each manager in the order
    the managers are declared.
    """

    @property
    def unmatched(self) -> list[str]:
        """
        Lists each hold naming no row of the table, in alphabetical order.
        """
        return sorted({*self.holds} - {lag.pin for lag in self.lags})

    def isolate(self) -> str | None:
        """
        Tries the lift again holding back each pin a major lift names, one
        pin at a time.

        Returns:
            The first such pin whose hold lets the rest of the lift pass its
            trial, or `None` where no single one does.
        """
        return next(
            (
                pin
                for pin in dict.fromkeys(lag.pin for lag in self.lifted() if lag.major)
                if self.passes({pin})
            ),
            None
        )

    def lifted(self, held: Collection[str] = ()) -> list[Lag]:
        """
        Keeps each lagging pin that neither `holds` nor `held` names.
        """
        return [lag for lag in self.lags if lag.pin not in {*self.holds, *held}]

    def passes(self, held: Collection[str] = ()) -> bool:
        """
        Writes the lift, short of each pin `held` names, into a scratch
        clone carrying the checkout's uncommitted and untracked files,
        rebuilds `.venv` there on the lifted interpreter, and runs `mise
        ci`, leaving the checkout as it was.

        Returns:
            Whether every step ran and `mise ci` passed.
        """
        with TemporaryDirectory(prefix="thermur-lift-") as folder:
            clone = self.checkout.clone(Path(folder))
            try:
                self.write(clone, held)
                clone.run("mise", "x", "--", "uv", "sync")
            except CalledProcessError:
                return False

            return clone.run("mise", "run", "ci", check=False).returncode == 0

    def table(self, now: datetime) -> str:
        """
        Writes every lagging pin as one Markdown table, each newest
        release's age counted in whole days up to `now`, or a sentence
        saying no pin lags where none does.
        """
        if not self.lags:
            return "Every pin stands at its newest release."

        return "\n".join(
            [
                "| **Pin** | **Declared In** | **Current** | **Newest** | **Major** "
                "| **Age in Days** | **Clears** |",
                "|---|---|---|---|---|---|---|",
                *(lag.row(now) for lag in self.lags)
            ]
        )

    def write(self, checkout: Checkout, held: Collection[str] = ()):
        """
        Writes the newest release of each pin the lift moves, short of each
        pin `held` names, wherever `checkout` declares it, then locks and
        installs the tools, upgrades each package of `uv.lock` the lift
        moves, and relocks every other lockfile through `lock:sync`.

        Raises:
            CalledProcessError : Where a command the write runs exits
                                 nonzero.
            ValueError         : Where a file names no release an edit
                                 lifts.
        """
        lifted = self.lifted(held)
        for lag in lifted:
            lag.write(checkout.root)

        checkout.run("mise", "lock")
        checkout.run("mise", "install")
        if upgraded := [
            *("--upgrade-package", lag.pin) for lag in lifted if lag.upgraded
        ]:
            checkout.run("mise", "x", "--", "uv", "lock", "--no-locked", *upgraded)

        checkout.run("mise", "run", "lock:sync")


class Manager(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One source of pins, whose report names each pin the checkout declares
    through it that lags its newest release.
    """

    checkout: Checkout
    """
    The clone the manager reads.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each pin the manager reports lagging, which a manager reading
        nothing of its own leaves empty.
        """
        return []

    def audit(self, *arguments: str | Path) -> Audit:
        """
        Reads the advisories `uv audit` reports on `uv.lock`, or on the
        lockfile of the script `arguments` names through `--script`, which
        uv prints whether it exits `0` or, on finding an advisory, `1`.

        Raises:
            CalledProcessError: Where `uv audit` exits above `1`, or exits
                                `1` printing no report, as it does where OSV
                                cannot be reached or the lockfile is stale.
        """
        audited = self.checkout.run(
            "uv",
            "audit",
            "--output-format",
            "json",
            "--preview-features",
            "audit-command,json-output",
            *arguments,
            capture = True,
            check   = False
        )
        if audited.returncode > 1 or not audited.stdout:
            audited.check_returncode()

        return Audit.model_validate_json(audited.stdout)

    def published(self, name: str, version: str) -> str | None:
        """
        Reads when the release `version` of the npm package `name` was
        published, as `bun info` reports it from the site's folder, or
        `None` where it reports no date.
        """
        return loads(
            self.read(
                "bun",
                "info",
                name,
                "time",
                "--json",
                folder = self.checkout.package.folder
            )
        ).get(version)

    def read(self, *command: str | Path, folder: Path = Path()) -> str:
        """
        Runs `command` in `folder` under the root and reads what it prints.

        Raises:
            CalledProcessError: Where the command exits nonzero.
        """
        return self.checkout.run(*command, capture=True, folder=folder).stdout

    def resolve(self, *requirements: str) -> Tree:
        """
        Resolves `requirements` through a script declaring them under the
        floor `requires-python` sets, written into a folder of its own.
        """
        with TemporaryDirectory(prefix="thermur-") as folder:
            script = Path(folder, "resolve.py")
            script.write_text(
                "# /// script\n"
                f"# dependencies    = {dumps(requirements)}\n"
                f'# requires-python = ">={self.checkout.manifest.floor}"\n'
                "# ///\n",
                encoding = "utf-8"
            )
            return self.tree(script, "--no-locked")

    def tree(self, script: Path, *flags: str) -> Tree:
        """
        Reads the packages `uv tree --format json` resolves under `flags`
        for the script at `script`.
        """
        return Tree.model_validate_json(
            self.read(
                "uv",
                "tree",
                "--format",
                "json",
                "--preview-features",
                "json-output",
                *flags,
                "--script",
                script
            )
        )

    def uploaded(self, name: str, version: str) -> datetime | None:
        """
        Reads when the first file of the release `version` of the package
        `name` was uploaded, or `None` where uv lists no upload time.
        """
        return self.resolve(f"{name}=={version}").package(name).uploaded


class Actions(Manager):
    """
    GitHub, reporting each action a workflow or a composite action pins by
    commit, against the commit the tag of its latest release names.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each pinned action whose latest release tags a commit other
        than its pin, naming the current release by the most specific tag
        naming the pinned commit, or by the commit's opening where no tag
        names it.
        """
        steps = self.checkout.steps
        return [
            Lag(
                current = max(
                    (tag for tag, tagged in tags.items() if tagged == commit),
                    default = commit[:7],
                    key     = len
                ),
                edits = tuple(
                    Edit(
                        ahead   = f"{escape(action)}@",
                        current = commit,
                        file    = path,
                        newest  = tags[release["tagName"]]
                    )
                    for path in dict.fromkeys(
                        path for path, step in steps if step.pin == (action, commit)
                    )
                ),
                newest    = release["tagName"],
                pin       = action,
                published = release["publishedAt"]
            )
            for action, commit in dict.fromkeys(
                step.pin for _, step in steps if step.pin
            )
            for release, tags in self.releases(action)
            if tags[release["tagName"]] != commit
        ]

    def releases(self, action: str) -> list[tuple[dict, dict[str, str]]]:
        """
        Reads the latest release of the repository holding `action`, named
        by its owner and name, through `gh release view`, and the commit
        each of its tags names through `git ls-remote`. A tag annotating
        a commit reads as that commit, which `git ls-remote` lists as
        `<tag>^{}` right after the tag.

        Returns:
            The release's `tagName` and `publishedAt` beside the commit each
            tag names, by the tag, as the one pair the list holds, or no
            pair where `gh` finds no release.

        Raises:
            CalledProcessError: Where `gh` fails for any other reason, or
                                `git` fails.
        """
        repository = "/".join(action.split("/")[:2])
        released   = self.checkout.run(
            "gh",
            "release",
            "view",
            "--json",
            "publishedAt,tagName",
            "--repo",
            repository,
            capture = True,
            check   = False
        )
        if "release not found" in released.stderr:
            return []

        released.check_returncode()
        return [
            (
                loads(released.stdout),
                {
                    tag.removesuffix("^{}"): commit
                    for commit, tag in findall(
                        r"(?m)^(\w+)\trefs/tags/(.+)$",
                        self.read(
                            "git",
                            "ls-remote",
                            "--tags",
                            f"https://github.com/{repository}"
                        )
                    )
                }
            )
        ]


class Builds(Manager):
    """
    uv, reporting the build requirement and each build constraint
    `pyproject.toml` pins exactly, against the releases uv resolves for all
    of them left unpinned together, so a constraint on a dependency of the
    build backend lifts only to a release the newest backend admits.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each build requirement and constraint whose package resolves
        to a release past its pin.
        """
        manifest = self.checkout.manifest
        pinned   = [*manifest.requires, *manifest.constraints]
        tree     = self.resolve(*(requirement.name for requirement in pinned))
        return [
            Lag.declared(
                {manifest.file: rf'"{escape(requirement.name)}=='},
                current   = current,
                newest    = resolved.version,
                pin       = requirement.name,
                published = resolved.uploaded
            )
            for requirement in pinned
            if (current := read_pin(requirement))
            and (resolved := tree.package(requirement.name))
            and resolved.version != current
        ]


class Images(Manager):
    """
    GitHub, reporting the runner image the jobs run on against the newest
    version of its family the README of `actions/runner-images` gives a
    `runs-on` label outside a preview. An image is named by its family
    and read as its version, so `ubuntu-26.04` is the release `26.04` of
    `ubuntu`.
    """

    label: ClassVar[str] = r"([a-z]+)-(\d[\d.]*)"
    """
    The pattern a runner image's label takes, its family and its version.
    """

    @property
    def images(self) -> dict[tuple[str, str], list[Path]]:
        """
        Maps the family and the version of each image a job runs on to every
        workflow naming it, in path order, leaving out a label holding more
        than a family and a version, such as one ending on `-arm`.
        """
        jobs = self.checkout.jobs
        return {
            named.groups(): [
                *dict.fromkeys(path for path, _, job in jobs if job.image == image)
            ]
            for image in dict.fromkeys(job.image for _, _, job in jobs)
            if (named := fullmatch(self.label, image))
        }

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each image whose family the README gives a newer version,
        lifted in every workflow naming it.
        """
        newest = self.newest
        return [
            Lag.declared(
                dict.fromkeys(
                    files,
                    rf"\bruns-on\s*:\s*(?:&[\w-]+\s+)?{escape(family)}-"
                ),
                current = version,
                newest  = lifted,
                pin     = family
            )
            for (family, version), files in self.images.items()
            if read_release(lifted := newest.get(family, version))
            > read_release(version)
        ]

    @property
    def newest(self) -> dict[str, str]:
        """
        Reads the newest version of each image family the table in the
        README of `actions/runner-images` gives a `runs-on` label, through
        `gh api`, leaving out each row a `preview` badge marks.
        """
        readme = self.read(
            "gh",
            "api",
            "--header",
            "Accept: application/vnd.github.raw+json",
            "repos/actions/runner-images/readme"
        )
        return dict(
            sorted(
                (
                    *findall(f"`{self.label}`", row)
                    for row in readme.splitlines()
                    if "preview" not in row
                ),
                key = lambda label: read_release(label[1])
            )
        )


class Locked(Manager):
    """
    uv, reporting each package `uv.lock` resolves that `uv lock --upgrade`
    moves, so a row names only a release the resolver reaches under every
    constraint the tree carries.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each package the dry run of the upgrade reports moving, beside
        each advisory `uv audit` reports that the move clears.
        """
        audit = self.audit()
        moves = findall(
            r"(?m)^Update (\S+) v(\S+) -> v(\S+)$",
            self.checkout.run(
                "uv",
                "lock",
                "--dry-run",
                "--no-locked",
                "--upgrade",
                capture = True
            ).stderr
        )
        return [
            Lag(
                advisories = audit.clears(current, name, newest),
                current    = current,
                newest     = newest,
                pin        = name,
                published  = self.uploaded(name, newest),
                upgraded   = True
            )
            for name, current, newest in moves
        ]


class Scripts(Manager):
    """
    uv, reporting each exact pin a task script's inline metadata declares
    against the newest release `uv tree --outdated` reports for it.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists the lagging pins of each task script declaring dependencies.
        """
        return [*self.pins(task) for task in self.checkout.tasks if task.dependencies]

    def pins(self, task: Task) -> list[Lag]:
        """
        Lists each exact pin the script of `task` declares whose package
        has a newer release, beside each advisory `uv audit` reports on the
        script that the lift clears.
        """
        audit = self.audit("--script", task.file)
        tree  = self.tree(task.file, "--depth", "0", "--outdated")
        return [
            Lag.declared(
                {
                    task.file.relative_to(self.checkout.root):
                        rf'"{escape(requirement.name)}=='
                },
                advisories = audit.clears(current, requirement.name, newest),
                current    = current,
                newest     = newest,
                pin        = requirement.name,
                published  = self.uploaded(requirement.name, newest)
            )
            for requirement in task.dependencies
            if (current := read_pin(requirement))
            and (resolved := tree.package(requirement.name))
            and (newest := resolved.latest)
        ]


class Site(Manager):
    """
    Bun, reporting each package `docs/package.json` pins that `bun outdated`
    lists, against the release its `latest` tag names.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each package `bun outdated` prints a row for, beside the
        advisories the lift clears on it.
        """
        package = self.checkout.package
        return self.cleared(
            [
            Lag.declared(
                {package.file: rf'"{escape(name)}"\s*:\s*"'},
                current   = current,
                newest    = newest,
                pin       = name,
                published = self.published(name, newest)
            )
            for name, current, newest in findall(
                r"(?m)^\| (\S+)(?: \(\w+\))? +\| (\d\S*) +\| \S+ +\| (\S+) +\|$",
                self.read("bun", "outdated", folder=package.folder)
            )
            ]
        )

    def advisories(self, checkout: Checkout) -> dict[str, set[str]]:
        """
        Reads each advisory `bun audit --json` reports on the site's
        packages in `checkout`, by the package it concerns, each named by
        the advisory identifier closing its address.
        """
        audited = checkout.run(
            "bun",
            "audit",
            "--json",
            capture = True,
            check   = False,
            folder  = self.checkout.package.folder
        )
        return {
            name: {advisory["url"].rpartition("/")[2] for advisory in advisories}
            for name, advisories in loads(audited.stdout or "{}").items()
        }

    def cleared(self, lags: list[Lag]) -> list[Lag]:
        """
        Copies each of `lags` with the advisories `bun audit` reports on
        its package that a lockfile resolved from the lifted pins no longer
        carries, auditing a copy of `docs/` only where an advisory names one
        of them.
        """
        package = self.checkout.package
        before  = self.advisories(self.checkout)
        if not before.keys() & {lag.pin for lag in lags}:
            return lags

        with TemporaryDirectory(prefix="thermur-") as scratch:
            copy = Checkout(root=scratch)
            self.checkout.copy([package.file, package.folder / "bun.lock"], copy.root)

            for lag in lags:
                lag.write(copy.root)

            copy.run(
                "bun",
                "install",
                "--lockfile-only",
                capture = True,
                folder  = package.folder
            )
            after = self.advisories(copy)

        return [
            lag.model_copy(
                update = {
                    "advisories": tuple(
                        sorted(before.get(lag.pin, set()) - after.get(lag.pin, set()))
                    )
                }
            )
            for lag in lags
        ]


class Tools(Manager):
    """
    mise, reporting each tool `.mise/config.toml` pins and the `min_version`
    floor it sets on mise itself, read as the release the aqua registry
    names `aqua:jdx/mise`. The `required-version` in `[tool.uv]` restates
    the `uv` pin, so a lift of `uv` writes both, and a lift of `python` to
    another minor moves each floor and restatement `repo:audit` holds to it.
    """

    mise: ClassVar[str] = "aqua:jdx/mise"
    """
    The name mise reports its own releases under.
    """

    @property
    def floors(self) -> dict[Path, str]:
        """
        Maps each file restating the minor of the `python` pin to the
        pattern the text ahead of that minor matches there, meaning
        the `requires-python` floor and the `target-version` in
        `pyproject.toml`, the README's badge and the Python it requires,
        and the `requires-python` of each task script whose inline metadata
        declares one.
        """
        manifest = self.checkout.manifest
        return {
            manifest.file   : r'(?:^requires-python\s*=\s*">=|^target-version\s*=\s*")',
            manifest.readme : r"(?:\[!\[Python |badge/python-|\*\*Python\*\*: )",
            **{
                task.file.relative_to(
                    self.checkout.root
                ): r'^# requires-python\s*=\s*">='
                for task in self.checkout.tasks
                if "requires-python" in task.metadata
            }
        }

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each tool `mise outdated --bump` reports a newer release of,
        then the floor where mise has released past it.
        """
        config = self.checkout.config
        return [
            *(
                Lag(
                    current   = tool["requested"],
                    edits     = self.edits(tool["requested"], tool["latest"], name),
                    newest    = tool["latest"],
                    pin       = name,
                    published = self.released(name, tool["latest"])
                )
                for name, tool in self.outdated("--local").items()
            ),
            *(
                Lag.declared(
                    {config.file: r'^min_version\s*=\s*"'},
                    current   = config.floor,
                    newest    = tool["latest"],
                    pin       = "mise",
                    published = self.released(self.mise, tool["latest"])
                )
                for tool in self.outdated(f"{self.mise}@{config.floor}").values()
            )
        ]

    def edits(self, current: str, newest: str, tool: str) -> tuple[Edit, ...]:
        """
        Lists each edit lifting the pin on `tool` from `current` to `newest`
        wherever `places` finds it, and, where a `python` lift moves the
        minor, each place `floors` names, from the minor of `current` to the
        minor of `newest`.
        """
        moves = [(self.places(tool), current, newest)]
        if tool == "python" and (floor := read_minor(current)) != (
            lifted := read_minor(newest)
        ):
            moves.append((self.floors, floor, lifted))

        return tuple(
            Edit(
                ahead   = ahead,
                current = old,
                file    = file,
                newest  = new
            )
            for places, old, new in moves
            for file, ahead in places.items()
        )

    def outdated(self, *arguments: str) -> dict[str, dict]:
        """
        Reads what `mise outdated --bump --json` reports for `arguments`,
        each tool it finds a newer release of by its name.
        """
        return loads(self.read("mise", "outdated", "--bump", "--json", *arguments))

    def places(self, tool: str) -> dict[Path, str]:
        """
        Maps each file declaring the pin on `tool` to the pattern the
        text ahead of its release matches there. In `.mise/config.toml`
        the pattern reaches past the opening of the inline table a
        tool takes for an option, and in `pyproject.toml` it finds the
        `required-version` restating the `uv` pin.
        """
        return {
            self.checkout.config.file: rf'^"?{escape(tool)}"?\s*=\s*'
            r'(?:\{[^}\n]*?\bversion\s*=\s*)?"',
            **(
                {self.checkout.manifest.file: r'^required-version\s*=\s*"=='}
                if tool == "uv"
                else {}
            )
        }

    def released(self, tool: str, version: str) -> str | None:
        """
        Reads when mise reports the release `version` of `tool` was
        published, or `None` where it reports no date.
        """
        return {
            remote["version"]: remote.get("created_at")
            for remote in loads(
                self.read("mise", "ls-remote", "--json", f"{tool}@{version}")
            )
        }.get(version)


class Wrangler(Manager):
    """
    Bun, reporting the release of wrangler each step installs through the
    `wranglerVersion` input of `cloudflare/wrangler-action`, against the
    release the `latest` tag of the `wrangler` package names.
    """

    @property
    def lags(self) -> list[Lag]:
        """
        Lists each release of wrangler a step names short of the newest,
        lifted in every file naming it.
        """
        pinned = [
            (path, str(version))
            for path, step in self.checkout.steps
            if (version := step.inputs.get("wranglerVersion")) is not None
        ]
        if not pinned:
            return []

        newest = self.read(
            "bun",
            "info",
            "wrangler",
            "version",
            folder = self.checkout.package.folder
        ).strip()
        return [
            Lag.declared(
                dict.fromkeys(
                    (path for path, named in pinned if named == current),
                    r"^\s*wranglerVersion\s*:\s*[\"']?"
                ),
                current   = current,
                newest    = newest,
                pin       = "wrangler",
                published = self.published("wrangler", newest)
            )
            for current in dict.fromkeys(named for _, named in pinned)
            if current != newest
        ]
