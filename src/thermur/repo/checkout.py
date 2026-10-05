"""
Holds `Checkout`, the clone the repository audit reads and the pipeline
resolves every path under, which keeps each document it validates for every
later check that reads it, lists the tasks and the programs the clone runs,
and reads the commit checked out and the digest of `uv.lock`.
"""

from functools          import cached_property
from hashlib            import sha256
from importlib.metadata import distribution
from pathlib            import Path
from pydantic           import AfterValidator, BaseModel, Field, TypeAdapter
from subprocess         import check_output
from typing             import Annotated, Self

from thermur.repo.schemas import Config, Manifest, Task


class Checkout(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The clone the audit reads and the pipeline resolves every path under,
    rooted at `root`, listing each program relative to that root and each
    task at the path mise resolves.
    """

    root: Annotated[Path, AfterValidator(Path.resolve)] = Field(
        default_factory = Path.cwd
    )
    """
    The folder holding `pyproject.toml`, resolved through any symlink, the
    working directory by default.
    """

    @property
    def commit(self) -> str:
        """
        Reads the commit checked out at the root, as `git rev-parse HEAD`
        prints it without its line break.
        """
        return check_output(
            ["git", "rev-parse", "HEAD"],
            cwd  = self.root,
            text = True
        ).strip()

    @cached_property
    def config(self) -> Config:
        """
        Validates `.mise/config.toml`.
        """
        return Config.read(self.root)

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

    @property
    def lockfile(self) -> str:
        """
        Hashes `uv.lock` with SHA-256, which pins the version of every
        library the package imports.
        """
        return sha256((self.root / "uv.lock").read_bytes()).hexdigest()

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
