"""
Holds `Checkout`, the clone the repository audit reads, which keeps each
document it validates for every later check that reads it and lists the
tasks and the programs the clone runs.
"""

from functools  import cached_property
from pathlib    import Path
from pydantic   import AfterValidator, BaseModel, Field, TypeAdapter
from subprocess import check_output
from typing     import Annotated

from thermur.repo.schemas import Config, Labels, Manifest, Task


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
    def scripts(self) -> list[Path]:
        """
        Lists every program in the folder the `_.path` of
        `.mise/config.toml` puts on the path, leaving out a dotfile such as
        the `.DS_Store` macOS writes into a folder Finder opens.
        """
        return sorted(
            path.relative_to(self.root)
            for path in (self.root / self.config.bin).iterdir()
            if not path.name.startswith(".")
        )

    @cached_property
    def tasks(self) -> list[Task]:
        """
        Keeps each task mise lists that a file under the root declares.
        """
        return [task for task in self.listed if task.source.is_relative_to(self.root)]

    def read(self, path: Path) -> str:
        """
        Reads the text of the file at `path` under the root.
        """
        return (self.root / path).read_text(encoding="utf-8")
