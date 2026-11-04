"""
Defines the fixture the task tests share, the loader that imports a task
script, described where it is defined.
"""

from collections.abc import Callable
from importlib.util  import module_from_spec, spec_from_file_location
from pathlib         import Path
from pytest          import Config, fixture
from types           import ModuleType


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
