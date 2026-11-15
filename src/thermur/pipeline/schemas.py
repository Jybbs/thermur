"""
Holds the records the pipeline reads and writes, meaning each `Subpackage`
whose source a digest node hashes and the `Run` a recorded run keeps beside
the tables its nodes derive.
"""

from ast       import Import, ImportFrom, parse, walk
from functools import cache
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
    def contents(self) -> set[Path]:
        """
        Lists every file under the folder, leaving out each `__pycache__`
        folder, which Python writes beside the source on a run outside mise.
        """
        return {
            path
            for path in self.folder.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }

    @property
    def name(self) -> str:
        """
        Reads the subpackage's name off its folder.
        """
        return self.folder.name

    @property
    def reached(self) -> set[Path]:
        """
        Lists every file the subpackage's code can run or read, meaning:

        - Each file under the folder
        - Each module of the package that one of those modules imports, and
          each module those modules import in turn
        - Each other file of a subpackage holding one of the added modules
        """
        root    = self.folder.parent
        reached = self.contents
        pending = [path for path in reached if path.suffix == ".py"]
        while pending:
            for module in self.imports(pending.pop()) - reached:
                holder   = Subpackage(folder=root / module.relative_to(root).parts[0])
                reached |= {module} | {
                    path for path in holder.contents if path.suffix != ".py"
                }
                pending.append(module)

        return reached

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
        Hashes every file `reached` lists beside `lock_digest`, the digest
        of `uv.lock`, so the result changes wherever any code the subpackage
        runs, a file it reads, or a library it imports changes, in whichever
        subpackage the code sits.

        A module hashes as Hamilton hashes a node's own source, leaving
        out every comment and each function's docstring while keeping the
        module's, each class's, and each field's docstring, and any other
        file hashes by its bytes.
        """
        return hash_value(
            (
                {
                    str(path.relative_to(self.folder, walk_up=True)): source_digest(
                        source = path.read_text(encoding="utf-8")
                    ) if path.suffix == ".py" else hash_value(path.read_bytes())
                    for path in self.reached
                },
                lock_digest
            )
        )

    def imports(self, module: Path) -> set[Path]:
        """
        Lists each module of the package that the module at `module`
        imports, by the file under the package's folder Python imports it
        from.
        """
        root  = self.folder.parent
        paths = {
            root.joinpath(*name.split(".")[1:]).with_suffix(".py")
            for name in imported_names(module.read_text(encoding="utf-8"))
            if name.startswith(f"{root.name}.")
        }
        return {path for path in paths if path.is_file()}

    @classmethod
    def listed(cls) -> list[Self]:
        """
        Lists every subpackage of the package, in name order.
        """
        folders = sorted(files("thermur").iterdir())
        return [cls(folder=folder) for folder in folders if folder.is_dir()]

    @classmethod
    def named(cls, name: str) -> Self:
        """
        Finds the subpackage called `name` in the folder Python imports the
        package from.
        """
        return cls(folder=files("thermur") / name)


@cache
def imported_names(source: str) -> frozenset[str]:
    """
    Names each module the Python `source` imports, both `a.b` from `import
    a.b` and `a.b` and `a.b.c` from `from a.b import c`, where `c` may
    itself be a module, keeping each answer under its `source` for the life
    of the process.
    """
    names = set[str]()
    for node in walk(parse(source)):
        match node:
            case Import(names=aliases): names |= {alias.name for alias in aliases}
            case ImportFrom(module=str(base), names=aliases):
                names |= {base} | {f"{base}.{alias.name}" for alias in aliases}

    return frozenset(names)


@cache
def source_digest(source: str) -> str:
    """
    Hashes the Python `source` as Hamilton hashes a node's own source,
    leaving out every comment and each function's docstring, keeping each
    answer under its `source` for the life of the process.
    """
    return hash_source_code(source, strip=True)
