"""
Defines the fixtures the task tests share, meaning the loader that imports
a task script, the folder of stand-in programs a task runs, and the snapshot
routed through plain text files, each described where it is defined.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from importlib.util   import module_from_spec, spec_from_file_location
from os               import pathsep
from pathlib          import Path
from pytest           import Config, MonkeyPatch, TempPathFactory, fixture
from shlex            import quote
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


@fixture
def stand_in(
    monkeypatch      : MonkeyPatch,
    tmp_path_factory : TempPathFactory
) -> Callable[[str, str], StandIn]:
    """
    Puts an empty folder ahead of every other on the path, so a program a
    case writes there answers in place of the real one.

    Returns:
        A writer taking the program's name and the `sh` lines it runs.
    """
    folder = tmp_path_factory.mktemp("bin")
    monkeypatch.setenv("PATH", str(folder), prepend=pathsep)

    def write(name: str, script: str) -> StandIn:
        """
        Writes an executable `sh` script named `name` into the folder, which
        logs each command it receives beside it before running the lines
        `script` holds.
        """
        program = StandIn(log=folder / f"{name}.log")
        logged  = quote(str(program.log))
        (folder / name).write_text(
            f"#!/bin/sh\nprintf '%s\\0' {quote(name)} \"$@\" >> {logged}\n"
            f"echo >> {logged}\n{script}\n",
            encoding = "utf-8"
        )
        (folder / name).chmod(0o755)
        return program

    return write
