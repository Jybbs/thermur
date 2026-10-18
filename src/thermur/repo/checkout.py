"""
Holds `Checkout`, the clone the `repo` tasks read and the pipeline resolves
every path under, which keeps each document it validates for every later
check that reads it. It lists the tasks, the programs, the workflows, the
composite actions, the issue templates, and the rulesets the clone holds
beside the checks each pull request runs, and the files git tracks there.
It reads the commit checked out and the digest of `uv.lock`, runs a command
from the root, and clones itself into a scratch folder.
"""

from collections.abc    import Iterable
from functools          import cached_property
from hashlib            import sha256
from importlib.metadata import distribution
from json               import loads
from os                 import environ, pathsep
from pathlib            import Path
from pydantic           import AfterValidator, BaseModel, Field, TypeAdapter
from re                 import sub
from subprocess         import CompletedProcess, run
from typing             import Annotated, Self

from thermur.repo.schemas import (
    Action,
    Citation,
    Config,
    Document,
    GithubSettings,
    Guide,
    Issue,
    Job,
    Labels,
    Lock,
    Manifest,
    Package,
    Release,
    Ruleset,
    Step,
    Task,
    Template,
    Workflow
)


class Checkout(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The clone the `repo` tasks read and the pipeline resolves every path
    under, rooted at `root`, naming each file relative to that root and each
    task at the path mise resolves.
    """

    root: Annotated[Path, AfterValidator(Path.resolve)] = Field(
        default_factory = Path.cwd
    )
    """
    The folder holding `pyproject.toml`, resolved through any symlink, the
    working directory by default.
    """

    @cached_property
    def actions(self) -> dict[Path, Action]:
        """
        Validates the manifest of each composite action under
        `.github/actions/`, by its path.
        """
        return self.documents(Action, ".github/actions/*/action.y*ml")

    @property
    def checks(self) -> list[tuple[str, str]]:
        """
        Lists each row of each workflow running on a pull request as the
        name the row's check run takes and the task the row runs, in the
        order the workflows and their rows declare them.
        """
        return [
            check
            for workflow in self.workflows.values()
            if workflow.on_pull_request
            for check in workflow.checks
        ]

    @cached_property
    def citation(self) -> Citation:
        """
        Validates `CITATION.cff`.
        """
        return Citation.read(self.root)

    @cached_property
    def commands(self) -> list[tuple[Path, str]]:
        """
        Pairs each file a task or a program on the path runs from, relative
        to the root, with the lines a shell runs out of it as `shell` reads
        them, where a task declared in TOML gives the lines it runs in place
        of a file's, read as `unwrap` reads them.
        """
        return [
            (
                task.path.relative_to(self.root),
                self.unwrap("\n".join(task.run)) if task.file is None else self.shell(
                    task.file
                )
            )
            for task in self.tasks
        ] + [(path, self.shell(self.root / path)) for path in self.scripts]

    @property
    def commit(self) -> str:
        """
        Reads the commit checked out at the root, as `git rev-parse HEAD`
        prints it without its line break.
        """
        return self.run("git", "rev-parse", "HEAD", capture=True).stdout.strip()

    @property
    def composites(self) -> dict[str, Path]:
        """
        Maps the name a step's `uses` gives each composite action, in the
        self-repository form, to the path of its manifest.
        """
        return {f"$/{path.parent}": path for path in self.actions}

    @cached_property
    def config(self) -> Config:
        """
        Validates `.mise/config.toml`.
        """
        return Config.read(self.root)

    @property
    def files(self) -> list[Path]:
        """
        Lists every file git tracks or would track under the root, relative
        to it, leaving out one gone from the disk while the index still
        lists it.
        """
        return [
            path
            for path in map(
                Path,
                self.run(
                    "git",
                    "ls-files",
                    "--cached",
                    "--exclude-standard",
                    "--others",
                    "-z",
                    capture = True
                ).stdout.split("\0")[:-1]
            )
            if (self.root / path).exists(follow_symlinks=False)
        ]

    @cached_property
    def github_settings(self) -> GithubSettings:
        """
        Validates `.github/settings.toml`.
        """
        return GithubSettings.read(self.root)

    @cached_property
    def guide(self) -> Guide:
        """
        Validates the tables of `.github/CONTRIBUTING.md`.
        """
        return Guide.read(self.root)

    @cached_property
    def issues(self) -> list[Issue]:
        """
        Reads every problem `mise tasks validate` reports on the tasks mise
        lists for the checkout, which it prints whatever status it exits
        with.
        """
        validated = self.run(
            "mise",
            "tasks",
            "validate",
            "--json",
            capture = True,
            check   = False
        )
        return TypeAdapter(list[Issue]).validate_python(
            loads(validated.stdout)["issues"]
        )

    @property
    def jobs(self) -> list[tuple[Path, str, Job]]:
        """
        Lists every job of every workflow beside the workflow's path and the
        job's key.
        """
        return [
            (path, key, job)
            for path, workflow in self.workflows.items()
            for key, job in workflow.jobs.items()
        ]

    @cached_property
    def labels(self) -> Labels:
        """
        Validates `.github/labels.toml`.
        """
        return Labels.read(self.root)

    @cached_property
    def listed(self) -> list[Task]:
        """
        Reads every task `mise tasks ls` lists for the checkout, hidden
        ones included and global ones left out, a task a file above the root
        declares among them.
        """
        return TypeAdapter(list[Task]).validate_json(
            self.run(
                "mise",
                "tasks",
                "ls",
                "--hidden",
                "--json",
                "--local",
                capture = True
            ).stdout
        )

    @cached_property
    def lock(self) -> Lock:
        """
        Validates `uv.lock`.
        """
        return Lock.read(self.root)

    @property
    def lock_digest(self) -> str:
        """
        Hashes `uv.lock` with SHA-256, which pins the version of every
        library the package imports.
        """
        return sha256((self.root / Lock.model_config["title"]).read_bytes()).hexdigest()

    @cached_property
    def manifest(self) -> Manifest:
        """
        Validates `pyproject.toml`.
        """
        return Manifest.read(self.root)

    @cached_property
    def package(self) -> Package:
        """
        Validates `docs/package.json`.
        """
        return Package.read(self.root)

    @cached_property
    def release(self) -> Release:
        """
        Validates `.github/release.yml`.
        """
        return Release.read(self.root)

    @property
    def rows(self) -> list[str]:
        """
        Lists the task each row of each workflow running on a pull request
        runs, in the order the workflows and their rows declare them.
        """
        return [task for _, task in self.checks]

    @cached_property
    def rulesets(self) -> dict[Path, Ruleset]:
        """
        Validates each ruleset under `.github/rulesets/`, by its path.
        """
        return self.documents(Ruleset, ".github/rulesets/*.json")

    @cached_property
    def scripts(self) -> list[Path]:
        """
        Lists every program in the folder the `_.path` of
        `.mise/config.toml` puts on the path.
        """
        return self.glob(f"{self.config.bin}/*")

    @property
    def steps(self) -> list[tuple[Path, Step]]:
        """
        Lists every step each composite action and each workflow job runs
        beside the path of the file declaring it, in path order.
        """
        return [
            (path, step)
            for path, holder in [
                *self.actions.items(),
                *((path, job) for path, _, job in self.jobs)
            ]
            for step in holder.steps
        ]

    @cached_property
    def tasks(self) -> list[Task]:
        """
        Keeps each task mise lists that a file under the root declares.
        """
        return [task for task in self.listed if task.source.is_relative_to(self.root)]

    @cached_property
    def templates(self) -> dict[Path, Template]:
        """
        Validates each issue template under `.github/ISSUE_TEMPLATE/`, by
        its path.
        """
        return self.documents(Template, ".github/ISSUE_TEMPLATE/*")

    @cached_property
    def workflows(self) -> dict[Path, Workflow]:
        """
        Validates each workflow under `.github/workflows/`, by its path.
        """
        return self.documents(Workflow, ".github/workflows/*.y*ml")

    def clone(self, folder: Path) -> Self:
        """
        Clones the checkout into the empty `folder` through `git clone
        --shared`, then copies in every file `files` lists and removes
        each file the clone holds that the checkout does not, so the clone
        carries the checkout's uncommitted and untracked files.

        Returns:
            The checkout rooted at the clone.
        """
        self.run("git", "clone", "--quiet", "--shared", self.root, folder)
        clone = type(self)(root=folder)
        for path in set(clone.files) - set(files := self.files):
            (clone.root / path).unlink()

        self.copy(files, clone.root)
        return clone

    def copy(self, files: Iterable[Path], folder: Path):
        """
        Copies each of `files`, named relative to the root, to the same path
        under `folder` in place of whatever that path holds, keeping each
        file's mode and a link as a link.
        """
        for path in files:
            (folder / path).parent.mkdir(exist_ok=True, parents=True)
            (folder / path).unlink(missing_ok=True)
            (self.root / path).copy(
                folder / path,
                follow_symlinks   = False,
                preserve_metadata = True
            )

    def documents[D: Document](self, kind: type[D], pattern: str) -> dict[Path, D]:
        """
        Validates each file under the root matching `pattern` as `kind`, by
        its path.
        """
        return {path: kind.read(self.root, path) for path in self.glob(pattern)}

    def glob(self, pattern: str) -> list[Path]:
        """
        Finds every file under the root matching `pattern`, relative to
        the root and in path order, leaving out a dotfile such as the
        `.DS_Store` macOS writes into a folder Finder opens.
        """
        return sorted(
            path.relative_to(self.root)
            for path in self.root.glob(pattern)
            if not path.name.startswith(".")
        )

    @classmethod
    def installed(cls) -> Self:
        """
        Roots a checkout at the clone `uv sync` installed the package from,
        which the `direct_url.json` that PEP 610 defines names, so every
        path under it resolves the same from any working directory.
        """
        return cls(root=Path.from_uri(distribution("thermur").origin.url))

    def read(self, path: Path) -> str:
        """
        Reads the text of the file at `path` under the root.
        """
        return (self.root / path).read_text(encoding="utf-8")

    def run(
        self,
        *command : str | Path,
        capture  : bool = False,
        check    : bool = True,
        folder   : Path = Path()
    ) -> CompletedProcess[str]:
        """
        Runs `command` in `folder` under the root, with mise trusting the
        root's configuration beside each one the environment already trusts,
        and without the `VIRTUAL_ENV` a task's `uv run` sets for the clone
        the task runs from, so uv reaches the root's own `.venv`.

        Args:
            capture : Whether the output is captured rather than printed.
            check   : Whether a nonzero exit raises.
            folder  : The folder to run in, relative to the root.

        Raises:
            CalledProcessError: Where `check` is set and the command exits
                                nonzero.
        """
        trusted = pathsep.join(
            filter(None, [environ.get("MISE_TRUSTED_CONFIG_PATHS"), str(self.root)])
        )
        return run(
            command,
            capture_output = capture,
            check          = check,
            cwd            = self.root / folder,
            env            = {
                name: value for name, value in environ.items() if name != "VIRTUAL_ENV"
            }
            | {"MISE_TRUSTED_CONFIG_PATHS": trusted},
            text = True
        )

    @staticmethod
    def shell(path: Path) -> str:
        """
        Reads the lines a shell runs out of the file at `path`, which for a
        Python task is the shebang alone and for any other file its lines as
        `unwrap` reads them.
        """
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            return text.partition("\n")[0]

        return Checkout.unwrap(text)

    @staticmethod
    def unwrap(text: str) -> str:
        """
        Reads every line of `text` a shell runs, blanking each line opening
        on `#` other than a shebang and joining each backslash continuation
        to the next line with a space.
        """
        return sub(r"(?m)^[ \t]*#(?!!).*$", "", text).replace("\\\n", " ")
