"""
Defines the fixtures the task tests share, the loader that imports a task
script, the copy of the `lock` tasks, and the `docs/` folder each `site`
task runs from, each described where it is defined.
"""

from collections.abc  import Callable
from common.edits     import executable
from common.stand_ins import StandIn
from importlib.util   import module_from_spec, spec_from_file_location
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture
from types            import ModuleType


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
def lock_copy(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn],
    tmp_path     : Path
) -> Path:
    """
    Copies the `lock` tasks, the library they source, `.mise/config.toml`,
    `.mise/mise.lock`, and `pyproject.toml` into `tmp_path`, keeping
    each file's mode, names the copy as the root a task sources its
    library from, clears the `AUDIT_IGNORE` list `mise test` sets from
    `.mise/config.toml`, and puts a stand-in `uv` that reports `uv.lock`
    current, a stand-in `bun` that reports `docs/bun.lock` current, and a
    stand-in `gh` that lists no patched release ahead on the path.

    Returns:
        The copy's root.
    """
    (tmp_path / ".mise/tasks").mkdir(parents=True)
    for path in (
        ".mise/config.toml", ".mise/lib", ".mise/mise.lock",
        ".mise/tasks/lock", "pyproject.toml"
    ):
        (pytestconfig.rootpath / path).copy(tmp_path / path, preserve_metadata=True)

    monkeypatch.delenv("AUDIT_IGNORE", raising=False)
    monkeypatch.setenv("MISE_PROJECT_ROOT", str(tmp_path))
    stand_in("bun", "exit 0")
    stand_in("gh", "exit 0")
    stand_in("uv", "exit 0")
    return tmp_path


@fixture
def site(tmp_path: Path) -> Path:
    """
    Writes a `docs/` folder holding an empty `.vitepress`, an empty `kit`,
    and a `node_modules/.bin` holding a stand-in for each tool a `site` task
    runs, each printing its own name and every argument it receives.

    Returns:
        The `docs/` folder, which every `site` task runs from.
    """
    for folder in (".vitepress", "kit"):
        (tmp_path / "docs" / folder).mkdir(parents=True)
    for tool in ("knip", "oxlint", "tsc", "vitepress", "vitest"):
        executable(
            tmp_path / "docs/node_modules/.bin" / tool,
            '#!/bin/sh\nprintf "%s\\n" "$(basename "$0")" "$@"\n'
        )

    return tmp_path / "docs"
