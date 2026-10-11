"""
Holds `Checkout`, the clone the repository audit and `repo:rulesets` read,
which keeps each document it validates for every later check that reads it
and lists the tasks, the programs, the workflows, the composite actions, the
issue templates, and the rulesets the clone holds, beside the checks each
pull request runs.
"""

from functools  import cached_property
from json       import loads
from pathlib    import Path
from pydantic   import AfterValidator, BaseModel, Field, TypeAdapter
from subprocess import check_output, run
from typing     import Annotated

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
    Release,
    Ruleset,
    Step,
    Task,
    Template,
    Workflow
)


class Checkout(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The clone the audit and `repo:rulesets` read, rooted at `root`, naming
    each file relative to that root and each task at the path mise resolves.
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
        validated = run(
            ["mise", "tasks", "validate", "--json"],
            capture_output = True,
            cwd            = self.root,
            text           = True
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
            check_output(
                ["mise", "tasks", "ls", "--hidden", "--json", "--local"],
                cwd  = self.root,
                text = True
            )
        )

    @cached_property
    def lock(self) -> Lock:
        """
        Validates `uv.lock`.
        """
        return Lock.read(self.root)

    @cached_property
    def manifest(self) -> Manifest:
        """
        Validates `pyproject.toml`.
        """
        return Manifest.read(self.root)

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

    def read(self, path: Path) -> str:
        """
        Reads the text of the file at `path` under the root.
        """
        return (self.root / path).read_text(encoding="utf-8")
