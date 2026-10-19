"""
Defines the fixtures the task tests share, meaning the loader that imports a
task script and the snapshot routed through plain text files, each described
where it is defined.
"""

from collections.abc  import Callable
from importlib.util   import module_from_spec, spec_from_file_location
from pathlib          import Path
from pytest           import Config, fixture
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from types import ModuleType


class PlainFile(SingleFileSnapshotExtension):
    """
    Writes each snapshot as a plain text file at
    `fixtures/<module>/<test>.txt` beside the tests that read it,
    the `fixtures` folder named by the `--snapshot-dirname` option in
    `[tool.pytest]`.
    """

    _write_mode    = WriteMode.TEXT
    file_extension = "txt"


@fixture(scope="session")
def load_task(pytestconfig: Config) -> Callable[[str], ModuleType]:
    """
    Imports a task script as a module, so a test calls what the script
    declares directly.

    Returns:
        A loader taking a path under the worktree root.
    """
    def load(path: str) -> ModuleType:
        """
        Imports the script at `path` under the worktree root as a module
        named for its file stem.
        """
        spec   = spec_from_file_location(Path(path).stem, pytestconfig.rootpath / path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """
    Routes every snapshot through the plain-file extension.
    """
    return snapshot.use_extension(PlainFile)
