"""
Pins the isolation the suite keeps from the machine running it, covering
the shell variables each test starts without, the home folder it reads, the
terminal type and width it runs under, the block that stops a test without
the `network` mark from opening a connection, the Unix socket a process
pool reaches its server over, the mark that lets a `network` test open a
connection, the files a `reads` mark leaves in a case's checkout, the shared
copy the read-only cases read apart from the worktree, build once, and read
through a `Checkout` holding no value an earlier case cached, the one JAX
compilation cache every worker shares, and the test modules a folder of the
same name would hide from collection.
"""

from collections.abc  import Iterator
from common.children  import runs
from common.edits     import committed
from common.isolation import CLEARED, copied_once, pytest_collection_modifyitems
from jax              import config
from os               import environ
from pathlib          import Path
from pytest           import (
    Config, FixtureRequest, MonkeyPatch, TempPathFactory, fixture, mark, raises, warns
)
from pytest_socket import SocketBlockedError
from socket        import create_connection, socketpair
from subprocess    import run
from sys           import executable

from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout

SETTING = Settings.model_config["env_prefix"] + next(iter(Settings.model_fields))


@fixture(
    params = [*CLEARED, SETTING.upper(), SETTING.lower(), "usage_try"],
    scope  = "module"
)
def name(request: FixtureRequest) -> Iterator[str]:
    """
    Sets the variable `request.param` names and yields that name.

    A module-scoped fixture runs before the function-scoped `environment`
    fixture, so the variable is set on every machine by the time
    `environment` clears it, a CI runner that never sets it included.
    """
    with MonkeyPatch.context() as patched:
        patched.setenv(request.param, "1")
        yield request.param


def test_a_network_test_gets_the_mark_that_opens_the_socket(request: FixtureRequest):
    """
    Asserts that the collection hook in `common.isolation` gives a test
    carrying the `network` mark pytest-socket's `enable_socket` mark, which
    lets that test open a connection.
    """
    request.node.add_marker(mark.network)
    pytest_collection_modifyitems([request.node])

    assert request.node.get_closest_marker("enable_socket")


def test_a_second_build_of_the_shared_copy_leaves_the_first_in_place(
    pytestconfig : Config,
    tmp_path     : Path
):
    """
    Asserts that building the shared copy where one already stands leaves
    each file of the first in place, so every worker after the first reads
    the copy the first built.
    """
    files = [Path("pyproject.toml")]
    first = copied_once(tmp_path / "checked", pytestconfig.rootpath, files)
    inode = (first.root / "pyproject.toml").stat().st_ino
    copied_once(tmp_path / "checked", pytestconfig.rootpath, files)

    assert (first.root / "pyproject.toml").stat().st_ino == inode


def test_a_socket_stays_closed_outside_the_network_mark():
    """
    Asserts that a test without the `network` mark cannot open a connection,
    pytest-socket issuing a warning and then raising `SocketBlockedError` on
    the attempt.

    `create_connection` looks `getaddrinfo` up on the socket module each
    time it runs rather than binding it once at import, and that name is the
    one pytest-socket replaces when a test starts.
    """
    with warns(UserWarning), raises(SocketBlockedError):
        create_connection(("blocked.invalid", 80))


def test_a_unix_socket_stays_open():
    """
    Asserts that a test can open a pair of connected Unix sockets, which
    `--allow-unix-socket` leaves open for a process pool reaching its server
    under the `forkserver` start method.

    `socketpair` builds each end through the `socket` class it looks up
    on the socket module when it runs, which is the class pytest-socket
    replaces, whereas a class bound at import would escape the block and
    pass without the option.
    """
    left, right = socketpair()
    with left, right:
        left.sendall(b"flock")
        assert right.recv(5) == b"flock"


@mark.reads("pyproject.toml", ".mise/**")
def test_a_reads_mark_copies_only_the_files_its_patterns_match(checkout: Checkout):
    """
    Asserts that a case carrying a `reads` mark finds in its checkout only
    the tracked files that match one of the mark's patterns.
    """
    copied = {
        path.relative_to(checkout.root).parts[0]
        for path in checkout.root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(checkout.root).parts
    }

    assert copied == {".mise", "pyproject.toml"}


def test_every_worker_shares_one_locked_jax_cache(pytestconfig: Config):
    """
    Asserts that JAX keeps each program it compiles under `.cache/jax` with
    a size limit set, under which it locks each entry through `filelock`, so
    every worker reads and writes the one cache at once.
    """
    assert config.jax_compilation_cache_dir == str(pytestconfig.rootpath / ".cache/jax")
    assert config.jax_compilation_cache_max_size != -1


def test_home_is_an_empty_folder():
    """
    Asserts that `~` resolves to an empty folder, so no test reads a dotfile
    of the developer's.
    """
    assert list(Path.home().iterdir()) == []


def test_no_test_module_shares_its_name_with_a_folder_beside_it(pytestconfig: Config):
    """
    Asserts that no module under `tests/` sits beside a folder of its own
    name, since under `--import-mode=importlib` pytest imports the namespace
    module it makes for the folder in the file's place and collects none of
    the file's cases.
    """
    tests = pytestconfig.rootpath / "tests"

    assert [path for path in tests.rglob("*.py") if path.with_suffix("").is_dir()] == []


def test_the_shared_copy_holds_each_file_apart_from_the_worktree(
    checked      : Checkout,
    pytestconfig : Config
):
    """
    Asserts that each file of the copy every read-only case shares holds an
    inode of its own, so no edit through it reaches the tracked file.
    """
    assert not (checked.root / "pyproject.toml").samefile(
        pytestconfig.rootpath / "pyproject.toml"
    )


@mark.slow
@mark.usefixtures("checked")
def test_the_shared_copy_holds_no_value_an_earlier_case_cached(
    pytestconfig     : Config,
    tmp_path         : Path,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that a case reading the shared copy after another case read a
    `cached_property` through it finds only the fields of `Checkout` set,
    in a suite of its own running the two in order on this conftest, so
    no earlier case hides a patch on `Checkout` from a later one whichever
    worker runs them. The suite carries an empty `pytest.toml`, so pytest
    reads no configuration from a folder above it. The case requests
    `checked` as well, so `py:affected` runs it on any change to that
    fixture, and calls `runs` on the modules declaring the records the inner
    suite reads, so a change to the top-level block of either runs it too.
    """
    root = pytestconfig.rootpath
    for path, text in {
        ".mise/config.toml"    : (root / ".mise/config.toml").read_text(),
        "pytest.toml"          : "",
        "tests/conftest.py"    : (root / "tests/conftest.py").read_text(),
        "tests/test_cached.py" : (
            "from thermur.repo.checkout import Checkout\n\n"
            "def test_caches(checked):\n"
            "    assert checked.config\n"
            "    assert 'config' in vars(checked)\n\n"
            "def test_starts_clean(checked):\n"
            "    assert vars(checked).keys() == Checkout.model_fields.keys()\n"
        )
    }.items():
        (tmp_path / path).parent.mkdir(exist_ok=True, parents=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    committed(Checkout(root=tmp_path))
    runs("thermur.repo.checkout", "thermur.repo.schemas")
    ran = run(
        [
            executable, "-m", "pytest", "--basetemp",
            tmp_path_factory.mktemp("inner"),
            "-p", "no:randomly", "-q", "tests"
        ],
        capture_output = True,
        cwd            = tmp_path,
        env            = {
            **environ,
            "PYTHONDONTWRITEBYTECODE" : "1",
            "PYTHONPATH"              : str(root / "tests")
        },
        text = True
    )

    assert ran.returncode == 0, ran.stdout


def test_the_shared_copy_is_built_once_for_every_worker(
    checked          : Checkout,
    tmp_path_factory : TempPathFactory,
    worker_id        : str
):
    """
    Asserts that the copy the read-only cases share sits in the folder
    holding every pytest-xdist worker's base temporary folder, so the run
    builds it once rather than once a worker.
    """
    base = tmp_path_factory.getbasetemp()

    assert checked.root.parent == (base if worker_id == "master" else base.parent)


def test_the_shell_carries_no_variable_that_changes_a_result(name: str):
    """
    Asserts that none of these variables reaches a test, covering the
    ones that set whether a console prints color and whether a task prints
    its GitHub Actions form, the files a GitHub Actions runner collects
    a workflow step's outputs and a workflow run's summary page from,
    the branch and the commit `gha:brief` prefers, the one adding options
    to each pytest run a case starts, the ones naming where a tool keeps
    its cache, configuration, data, and state, a variable opening on the
    `env_prefix` of `Settings` in either case, which sets a setting, and one
    opening on `usage_`, through which mise hands a task its flags.
    """
    assert name not in environ


@mark.parametrize(("variable", "value"), [("COLUMNS", "80"), ("TERM", "dumb")])
def test_the_terminal_reads_one_type_and_width_on_every_machine(
    value    : str,
    variable : str
):
    """
    Asserts that each test runs under the `dumb` terminal 80 columns wide,
    so a program sizing or coloring its output for the terminal prints alike
    on every machine.
    """
    assert environ[variable] == value
