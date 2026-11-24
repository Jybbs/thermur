"""
Pins the `thermur` command line, meaning the version it reports, the help
it prints for itself, for each group, and for each command, and that neither
the version, the help of either group, nor the fetch imports JAX or xarray.
It also pins that `spawned`, which those import cases run their children
through, passes `runs` each `thermur` module its child loaded and returns
only the lines the script printed.
"""

from collections.abc    import Callable
from common             import children
from common.children    import spawned
from importlib          import import_module, reload
from importlib.metadata import version
from pytest             import CaptureFixture, MonkeyPatch, fixture, mark, param, raises
from syrupy.assertion   import SnapshotAssertion


@fixture
def printed(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch
) -> Callable[..., str]:
    """
    Loads `thermur.cli` afresh, so no command an earlier case ran has
    resolved its import path, and sets the console's width to 80 columns,
    whatever the shell running the suite sets.

    Returns:
        A runner taking the tokens of one call of the command line, which it
        runs in this process, returning what the call prints.
    """
    monkeypatch.setenv("COLUMNS", "80")
    app = reload(import_module("thermur.cli")).app

    def call(*tokens: str) -> str:
        """
        Runs the command line on `tokens` and reads what it prints.
        """
        app(tokens, result_action="return_value")
        return capsys.readouterr().out

    return call


def test_a_spawned_script_runs_the_modules_its_child_loaded(monkeypatch: MonkeyPatch):
    """
    Asserts that `spawned` calls `runs` on a `thermur` module its child
    loaded and returns only the lines the script printed.
    """
    received = []
    monkeypatch.setattr(children, "runs", lambda *modules: received.extend(modules))
    assert spawned("from thermur.repo import checkout\nprint(checkout.__name__)") == [
        "thermur.repo.checkout"
    ]
    assert "thermur.repo.checkout" in received


def test_neither_the_version_nor_a_group_s_help_imports_jax_or_xarray():
    """
    Asserts that a fresh interpreter printing the version, and the help of
    the cut and the fetch groups and of their commands, imports neither JAX
    nor xarray, which load only once a command runs.
    """
    script = "; ".join(
        [
            "import sys", "from thermur.cli import app",
            *(
                f"app({tokens!r}, result_action='return_value')"
                for tokens in (
                    ["--version"],
                    ["cut", "--help"],
                    ["cut", "sample", "--help"],
                    ["fetch", "--help"],
                    ["fetch", "sample", "--help"]
                )
            ),
            "print(sorted({'jax', 'xarray'} & set(sys.modules)))"
        ]
    )

    assert spawned(script)[-1] == "[]"


def test_running_the_fetch_imports_neither_jax_nor_xarray():
    """
    Asserts that a fresh interpreter running the fetch imports neither JAX
    nor xarray, with the download itself stood in for.
    """
    script = "; ".join(
        [
            "import sys", "from thermur.cli import app",
            "from thermur.environment import schemas",
            "schemas.Sample.fetch = lambda self, root: root / self.path",
            "app(['fetch', 'sample'], result_action='return_value')",
            "print(sorted({'jax', 'xarray'} & set(sys.modules)))"
        ]
    )

    assert spawned(script)[-1] == "[]"


@mark.parametrize(
    "tokens",
    [
        param(("--help",), id="thermur"),
        param(("cut", "--help"), id="cut"),
        param(("cut", "sample", "--help"), id="cut-sample"),
        param(("fetch", "--help"), id="fetch"),
        param(("fetch", "sample", "--help"), id="fetch-sample"),
        param(("fly", "--help"), id="fly"),
        param(("watch", "--help"), id="watch")
    ]
)
def test_each_level_prints_its_help(
    printed  : Callable[..., str],
    snapshot : SnapshotAssertion,
    tokens   : tuple[str, ...]
):
    """
    Asserts that the help the command line, each group, and each command
    print reads as the snapshot holds it, each rendered from the summary and
    the docstrings the package declares.
    """
    assert printed(*tokens) == snapshot


@mark.parametrize(
    "tokens",
    [
        param((), id="thermur"),
        param(("cut",), id="cut"),
        param(("fetch",), id="fetch")
    ]
)
def test_a_level_given_no_command_prints_its_help(
    printed : Callable[..., str],
    tokens  : tuple[str, ...]
):
    """
    Asserts that the command line, or a group, given no command prints what
    `--help` prints at that level.
    """
    assert printed(*tokens) == printed(*tokens, "--help")


@mark.parametrize(
    ("tokens", "offered"),
    [
        param(("bogus",), "cut, fetch, fly, watch", id="thermur"),
        param(("cut", "bogus"), "sample", id="cut"),
        param(("fetch", "bogus"), "sample", id="fetch")
    ]
)
def test_an_unknown_command_exits_naming_the_commands_offered(
    capsys  : CaptureFixture[str],
    offered : str,
    printed : Callable[..., str],
    tokens  : tuple[str, ...]
):
    """
    Asserts that a token naming no command exits with status 2, and prints
    to the error stream the token beside each command that level offers.
    """
    with raises(SystemExit) as exited:
        printed(*tokens)

    assert exited.value.code == 2
    assert f'Unknown command "bogus". Available commands: {offered}.' in (
        capsys.readouterr().err
    )


def test_the_version_is_the_installed_package_s(printed: Callable[..., str]):
    """
    Asserts that `thermur --version` prints the version the installed
    package's metadata carries.
    """
    assert printed("--version") == f"{version('thermur')}\n"
