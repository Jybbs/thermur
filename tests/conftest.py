"""
Defines the fixtures every test folder shares, each described where it is
defined, beside the plain-file extension every snapshot is written through.
The autouse `environment` fixture isolates every test from the machine
running it, the collection hook lets a test open a network connection only
when it carries the `network` mark, and the `--share` option runs one share
of a suite split by test across several machines.
"""

from collections.abc    import Callable, Iterator
from common.edits       import executable, git
from common.environment import CLEARED
from common.opened      import Opened
from common.stand_ins   import StandIn
from os                 import environ, pathsep
from pathlib            import Path
from pytest             import (
    Config,
    FixtureRequest,
    Item,
    MonkeyPatch,
    Parser,
    TempPathFactory,
    fixture,
    mark
)
from shlex            import quote
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from tomllib import loads as from_toml

from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout
from thermur.repo.schemas      import read_pin


class PlainFile(SingleFileSnapshotExtension):
    """
    Writes each snapshot as a plain text file at
    `fixtures/<module>/<test>.txt` beside the tests that read it,
    the `fixtures` folder named by the `--snapshot-dirname` option in
    `[tool.pytest]`.
    """

    _write_mode    = WriteMode.TEXT
    file_extension = "txt"


@fixture
def checkout(
    monkeypatch  : MonkeyPatch,
    opens        : Opened,
    pytestconfig : Config,
    tmp_path     : Path,
    tracked      : list[Path]
) -> Checkout:
    """
    Copies into `tmp_path` every file `tracked` lists, keeping each
    program's mode, into a git repository holding no commit, so git lists
    each copied file as one it would track. It then trusts the copy's mise
    configuration and returns the `Checkout` rooted there, which a case
    edits before a check reads it. `opens` records nothing while the copy is
    made, and records each copied file the case opens under the path it was
    copied from.
    """
    copy = Checkout(root=tmp_path)
    with monkeypatch.context() as copying:
        copying.setattr(opens, "paused", True)
        Checkout(root=pytestconfig.rootpath).copy(tracked, copy.root)
    monkeypatch.setitem(opens.copies, tmp_path, pytestconfig.rootpath)
    copy.run("git", "init", "--quiet")
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(tmp_path))
    return copy


@fixture
def clone(checkout: Checkout) -> Checkout:
    """
    Commits the copy `checkout` makes into the git repository holding it, so
    a case reads the commit checked out there.
    """
    for command in (["add", "--all"], ["commit", "--message=Copy", "--quiet"]):
        git(checkout, *command)

    return checkout


@fixture(autouse=True)
def environment(
    monkeypatch      : MonkeyPatch,
    pytestconfig     : Config,
    tmp_path_factory : TempPathFactory
):
    """
    Clears every shell variable `CLEARED` names, any of which would let
    the machine running the suite change a result. Some set whether a
    console prints color, `GITHUB_ACTIONS` sets whether a task prints its
    GitHub Actions form, `GITHUB_OUTPUT` and `GITHUB_STEP_SUMMARY` name the
    files a GitHub Actions runner collects a workflow step's outputs and
    a workflow run's summary page from, and `GITHUB_HEAD_REF` and `COMMIT`
    name the branch and the commit `gha:brief` prefers over the ones a case
    sets. `CI`, which a GitHub Actions runner sets, and `MISE_YES` each
    turn on mise's `yes` setting, which answers a task's `confirm` prompt
    without asking. The rest name the folders where a tool keeps its cache,
    configuration, data, and state, which the XDG convention defines. It
    also clears every variable opening on the `env_prefix` of `Settings`,
    which pydantic-settings matches to the field it names whatever its case,
    and every one opening on `usage_`, through which mise hands a task each
    flag its usage spec declares and which every program the task starts
    inherits. It then points `HOME` at an empty folder the test owns, so a
    tool that falls back to `~` finds none of the developer's files there.

    mise reads a configuration file out of every folder above the one
    it runs in, as well as the file at `HOME`. Moving `HOME` leaves the
    developer's own configuration above the checkout as an ordinary ancestor
    rather than as the global file, and mise stops on it as one it was never
    told to trust. The two `MISE_` variables name that file as one to skip
    and this checkout as one to trust, so a test can run mise over it.
    """
    prefixes = (Settings.model_config["env_prefix"].lower(), "usage_")
    for name in [
        *CLEARED,
        *(name for name in environ if name.lower().startswith(prefixes))
    ]:
        monkeypatch.delenv(name, raising=False)

    settled = Path.home() / ".config" / "mise" / "config.toml"

    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setenv("MISE_IGNORED_CONFIG_PATHS", str(settled))
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(pytestconfig.rootpath))


@fixture
def mise(request: FixtureRequest, stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `mise` on the path running the `sh` lines a case passes
    through `request.param`, or exiting `0` where it passes none.
    """
    return stand_in("mise", getattr(request, "param", "exit 0"))


@fixture(autouse=True, scope="session")
def opens(pytestconfig: Config, tracked: list[Path]) -> Opened:
    """
    Adds the `Opened` hook over the files `tracked` lists where
    pytest-testmon collects, so a change to a file a test opens selects
    that test.
    """
    return Opened.added(pytestconfig, tracked)


@fixture(autouse=True)
def recorded(opens: Opened, request: FixtureRequest) -> Iterator[None]:
    """
    Records each tracked file the test opened once it ends, through
    `Opened.record`, those the setup of a fixture it requests opened
    included, whichever test ran that setup.
    """
    yield
    opens.record(request.fixturenames)


@fixture(scope="session")
def pins(pytestconfig: Config) -> dict[str, str]:
    """
    Reads the release each pin a case edits holds in the worktree, which
    the copy `checkout` makes holds as well, by the name a case formats into
    the text it edits and the message it expects, so no case names a release
    `lock:lift` moves.

    Returns:
        The `actions/checkout` commit, the minor `requires-python` sets as
        the floor, the runner image the first job runs on, the `minijinja`
        release `uv.lock` resolves, the `node` major, `python`, and `uv`
        mise pins, the `@types/node`, `knip`, and `vitest` releases the
        site pins, the `glob` release its `overrides` holds, the
        `@zag-js/toggle` release the kit pins as `toggle`, the
        `[project].version` the manifest declares, and each build requirement
        and constraint, by its package's name.
    """
    worktree = Checkout(root=pytestconfig.rootpath)
    manifest = worktree.manifest
    site     = worktree.dependencies[worktree.package.file].dev
    tools    = from_toml(worktree.read(Path(".mise/config.toml")))["tools"]
    kit      = worktree.dependencies[Path("docs/kit/package.json")].dependencies
    return {
        "checkout"  : dict(step.pin for _, step in worktree.steps if step.pin)[
            "actions/checkout"
        ],
        "floor"     : manifest.floor,
        "glob"      : worktree.package.overrides["glob"],
        "image"     : next(job for _, _, job in worktree.jobs).image,
        "knip"      : site["knip"],
        "minijinja" : worktree.lock.versions["minijinja"],
        "node"      : worktree.config.node,
        "python"    : tools["python"],
        "toggle"    : kit["@zag-js/toggle"],
        "types"     : site["@types/node"],
        "uv"        : tools["uv"],
        "version"   : manifest.version,
        "vitest"    : site["vitest"],
        **{
            requirement.name: read_pin(requirement)
            for requirement in [*manifest.requires, *manifest.constraints]
        }
    }


@fixture
def stand_in(
    monkeypatch      : MonkeyPatch,
    tmp_path_factory : TempPathFactory
) -> Callable[[str, str], StandIn]:
    """
    Puts an empty folder ahead of every other on the path, so a program a
    case writes there answers in place of the real one.

    Returns:
        A writer taking the program's name and the `sh` lines it runs, and
        returning the `StandIn` whose log holds each command it receives.
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
        executable(
            folder / name,
            f"#!/bin/sh\nprintf '%s\\0' {quote(name)} \"$@\" >> {logged}\n"
            f"echo >> {logged}\n{script}\n"
        )
        return program

    return write


@fixture(scope="session")
def tracked(pytestconfig: Config) -> list[Path]:
    """
    Lists every file of the worktree git tracks or would track, as
    `Checkout.tracked` lists them.
    """
    return Checkout(root=pytestconfig.rootpath).tracked()


def pytest_addoption(parser: Parser):
    """
    Adds `--share`, which takes `<index>/<count>` and runs the share of the
    suite at `index`, counting from 1, out of `count` shares.
    """
    parser.addoption(
        "--share",
        help    = "Run the share at <index>, counting from 1, out of <count> shares",
        metavar = "<index>/<count>",
        type    = share
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]):
    """
    Adds pytest-socket's `enable_socket` mark, which reopens the socket
    `--disable-socket` closes, to every test carrying the `network` mark,
    and skips every test outside the share `--share` names. A share takes
    every `count`-th test from the one at `index` in node ID order, which
    every machine computes alike under any `pytest-randomly` seed. A skip
    leaves syrupy reading every test as selected, so it still reports a
    snapshot no test reads, while it reads a skipped test's snapshots as
    still in use.
    """
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(mark.enable_socket)

    if picked := config.getoption("share"):
        index, count = picked
        kept         = set(sorted(item.nodeid for item in items)[index - 1 :: count])
        skip         = mark.skip(reason=f"Runs in another share than {index}/{count}")
        for item in items:
            if item.nodeid not in kept:
                item.add_marker(skip)


def share(text: str) -> tuple[int, int]:
    """
    Reads `<index>/<count>` as the share at `index` out of `count` shares.

    Raises:
        ValueError: Where `text` is no such pair, or `index` falls outside 1
                    to `count`.
    """
    index, count = map(int, text.split("/"))
    if not 1 <= index <= count:
        raise ValueError(f"No share {index} out of {count}")

    return index, count


@fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """
    Routes every snapshot through the plain-file extension.
    """
    return snapshot.use_extension(PlainFile)
