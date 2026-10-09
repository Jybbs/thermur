"""
Holds `Checkout`, the clone the repository audit reads, which keeps each
document it validates for every later check that reads it and lists the
tasks, the programs, and the issue templates the clone holds.
"""

from functools  import cached_property
from pathlib    import Path
from pydantic   import AfterValidator, BaseModel, Field, TypeAdapter
from subprocess import check_output
from typing     import Annotated

from thermur.repo.schemas import (
    Citation, Config, Labels, Manifest, Release, Task, Template
)


class Checkout(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The clone the audit reads, rooted at `root`, listing each program
    relative to that root and each task at the path mise resolves.
    """

    root: Annotated[Path, AfterValidator(Path.resolve)] = Field(
        default_factory = Path.cwd
    )
    """
    The folder holding `pyproject.toml`, resolved through any symlink, the
    working directory by default.
    """

    @cached_property
    def citation(self) -> Citation:
        """
        Validates `CITATION.cff`.
        """
        return Citation.read(self.root)

    @cached_property
    def config(self) -> Config:
        """
        Validates `.mise/config.toml`.
        """
        return Config.read(self.root)

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

    @cached_property
    def scripts(self) -> list[Path]:
        """
        Lists every program in the folder the `_.path` of
        `.mise/config.toml` puts on the path.
        """
        return self.glob(f"{self.config.bin}/*")

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
        return {
            path: Template.read(self.root, path)
            for path in self.glob(".github/ISSUE_TEMPLATE/*")
        }

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
