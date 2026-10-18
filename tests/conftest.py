"""
Defines the fixtures every test folder shares, each described where it is
defined. The autouse `environment` fixture isolates every test from the
machine running it, and the collection hook lets a test open a network
connection only when it carries the `network` mark.
"""

from collections.abc    import Callable
from common.edits       import executable
from common.environment import CLEARED
from common.stand_ins   import StandIn
from json               import loads
from os                 import environ, pathsep
from pathlib            import Path
from pytest             import (
    Config,
    FixtureRequest,
    Item,
    MonkeyPatch,
    TempPathFactory,
    fixture,
    mark
)
from shlex   import quote
from tomllib import loads as from_toml

from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout
from thermur.repo.schemas      import read_pin


@fixture
def checkout(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    tmp_path     : Path,
    tracked      : list[Path]
) -> Checkout:
    """
    Copies into `tmp_path` every file `tracked` lists, keeping each
    program's mode, then trusts the copy's mise configuration and returns
    the `Checkout` rooted there, which a case edits before a check reads it.
    """
    Checkout(root=pytestconfig.rootpath).copy(tracked, tmp_path)
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(tmp_path))
    return Checkout(root=tmp_path)


@fixture
def clone(checkout: Checkout) -> Checkout:
    """
    Commits the copy `checkout` makes into a git repository of its own, so a
    case reads the commit checked out there.
    """
    for command in (
        ["init", "--quiet"],
        ["add", "--all"],
        ["commit", "--message=Copy", "--quiet"]
    ):
        checkout.run(
            "git",
            "-c",
            "user.email=thermur@example.com",
            "-c",
            "user.name=Thermur",
            *command
        )

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


@fixture(scope="session")
def pins(pytestconfig: Config) -> dict[str, str]:
    """
    Reads the release each pin a case edits holds in the worktree, which
    the copy `checkout` makes holds as well, by the name a case formats into
    the text it edits and the message it expects, so no case names a release
    `lock:lift` moves.

    Returns:
        The `actions/checkout` commit, the minor `requires-python` sets as
        the floor, the `minijinja` release `uv.lock` resolves, the `node`
        major, `python`, and `uv` mise pins, the `@types/node` and `vitest`
        releases the site pins, and each build requirement and constraint,
        by its package's name.
    """
    worktree = Checkout(root=pytestconfig.rootpath)
    manifest = worktree.manifest
    site     = loads(worktree.read(Path("docs/package.json")))["devDependencies"]
    tools    = from_toml(worktree.read(Path(".mise/config.toml")))["tools"]
    return {
        "checkout"  : dict(step.pin for _, step in worktree.steps if step.pin)[
            "actions/checkout"
        ],
        "floor"     : manifest.floor,
        "minijinja" : worktree.lock.versions["minijinja"],
        "node"      : worktree.config.node,
        "python"    : tools["python"],
        "types"     : site["@types/node"],
        "uv"        : tools["uv"],
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
    Lists every file of the worktree git tracks or would track, relative to
    the root, leaving out one moved away before git learns of it, since the
    index still lists it.
    """
    return Checkout(root=pytestconfig.rootpath).files


def pytest_collection_modifyitems(items: list[Item]):
    """
    Adds pytest-socket's `enable_socket` mark to every test carrying the
    `network` mark. The `--disable-socket` option in `addopts` blocks every
    test from opening a connection, and a test carrying `enable_socket` can
    reach a live service again.
    """
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(mark.enable_socket)
