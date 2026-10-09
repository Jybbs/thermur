"""
Holds the records the pipeline reads and writes, meaning each `Subpackage`
whose source a digest node hashes and the `Run` a recorded run keeps beside
the tables its nodes derive.
"""

from hamilton.caching.fingerprinting import hash_value
from hamilton.graph_types            import hash_source_code
from importlib.resources             import files
from pathlib  import Path
from platform import platform
from pydantic import BaseModel, Field, StringConstraints
from typing   import Annotated, Self

from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout

type Hexadecimal = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]+$")]


class Run(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    What a recorded run keeps beside the tables its nodes derive, meaning
    the settings every node read and the code and the platform that computed
    them.
    """

    commit: Hexadecimal
    """
    The commit checked out in the clone, as `git rev-parse HEAD` prints it.
    """

    lock_digest: Hexadecimal
    """
    The SHA-256 digest of the clone's `uv.lock`, which pins the version of
    every library the run imported.
    """

    settings: Settings
    """
    The settings every node read.
    """

    platform: str = Field(default_factory=platform)
    """
    The platform the run computed on, as `platform.platform` names it.
    """

    @classmethod
    def from_checkout(cls, checkout: Checkout, settings: Settings) -> Self:
        """
        Pairs `settings` with the commit `checkout` has checked out and the
        digest of its `uv.lock`, on the platform running the call.
        """
        return cls(
            commit      = checkout.commit,
            lock_digest = checkout.lock_digest,
            settings    = settings
        )


class Subpackage(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One subpackage of the package, at the folder Python imports it from.
    """

    folder: Path
    """
    The folder holding the subpackage's modules and every file they read.
    """

    @property
    def name(self) -> str:
        """
        Reads the subpackage's name off its folder.
        """
        return self.folder.name

    @property
    def steps(self) -> str | None:
        """
        Names the dotted module holding the subpackage's Hamilton steps.

        Returns:
            The module's dotted name, or `None` where the subpackage holds
            no `steps.py`.
        """
        if (self.folder / "steps.py").is_file():
            return f"thermur.{self.name}.steps"

        return None

    def digest(self, lock_digest: str) -> str:
        """
        Hashes every file under the folder beside `lock_digest`, the digest
        of `uv.lock`, so the result changes wherever the subpackage's code,
        a file it reads, or a library it imports does.

        A module hashes as Hamilton hashes a node's own source, leaving
        out every comment and each function's docstring while keeping the
        module's, each class's, and each field's docstring, and any other
        file hashes by its bytes. A `__pycache__` folder is left out, since
        Python writes it beside the source on a run outside mise.
        """
        return hash_value(
            (
                {
                    str(path.relative_to(self.folder)): hash_source_code(
                        path.read_text(encoding="utf-8"),
                        strip = True
                    ) if path.suffix == ".py" else hash_value(path.read_bytes())
                    for path in self.folder.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                },
                lock_digest
            )
        )

    @classmethod
    def listed(cls) -> list[Self]:
        """
        Lists every subpackage of the package, in name order.
        """
        folders = sorted(files("thermur").iterdir())
        return [cls(folder=folder) for folder in folders if folder.is_dir()]
