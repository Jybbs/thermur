"""
Holds the isolation every test keeps from the machine running it beside
the autouse `environment` fixture, meaning the shell variables each test
starts without besides those opening on the `env_prefix` of `Settings`,
the terminal type and width it runs under, the collection hook that lets
a test open a network connection only when it carries the `network` mark,
and the build of the shared copy the read-only cases read apart from the
worktree. The `-p` option in `addopts` loads this module as a plugin, and
`tests/isolation.py` checks it.
"""

from filelock import FileLock
from pathlib  import Path
from pytest   import Item, mark

from thermur.repo.checkout import Checkout

CLEARED = (
    "CI", "CLICOLOR", "CLICOLOR_FORCE", "COLORTERM", "COMMIT",
    "FORCE_COLOR", "GITHUB_ACTIONS", "GITHUB_HEAD_REF", "GITHUB_OUTPUT",
    "GITHUB_STEP_SUMMARY", "MISE_OFFLINE", "MISE_YES", "NO_COLOR",
    "PYTEST_ADDOPTS", "UV_OFFLINE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "XDG_DATA_HOME", "XDG_STATE_HOME"
)

TERMINAL = {"COLUMNS": "80", "TERM": "dumb"}


def copied_once(root: Path, source: Path, tracked: list[Path]) -> Checkout:
    """
    Copies each file of `tracked` from `source` into a git repository
    holding no commit at `root`, unless a build there came first, under a
    lock beside `root` that every pytest-xdist worker takes in turn.
    """
    copy = Checkout(root=root)
    with FileLock(root.with_suffix(".lock")):
        if not root.exists():
            Checkout(root=source).copy(tracked, root)
            copy.run("git", "init", "--quiet")

    return copy


def pytest_collection_modifyitems(items: list[Item]):
    """
    Adds pytest-socket's `enable_socket` mark, which reopens the socket
    `--disable-socket` closes, to every test carrying the `network` mark.
    """
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(mark.enable_socket)
