"""
Pins that the install runs Bun under the frozen lockfile and then prunes,
skipping both while `docs/package.json` and `docs/bun.lock` are unchanged
and `docs/node_modules/.installed` exists.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture, mark
from subprocess       import CompletedProcess, run
from tomllib          import loads as from_toml

INSTALLED = ["bun install --frozen-lockfile", "bun prune"]


@fixture
def bun(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    site         : Path,
    stand_in     : Callable[[str, str], StandIn]
) -> StandIn:
    """
    Copies `.mise/config.toml` and the install task above `site`, keeping
    the task's mode, and trusts the copy, turning off each tool it pins,
    which mise would otherwise install into the empty `HOME` before running
    the task. It then writes a `package.json` and a `bun.lock` into `site`
    and puts a stand-in `bun` on the path that writes `node_modules` where
    it runs.
    """
    (site.parent / ".mise/tasks/site").mkdir(parents=True)
    for path in (".mise/config.toml", ".mise/tasks/site/install"):
        (pytestconfig.rootpath / path).copy(
            site.parent / path, preserve_metadata=True
        )
    for name in ("package.json", "bun.lock"):
        (site / name).write_text("{}\n", encoding="utf-8")

    config = (site.parent / ".mise/config.toml").read_text(encoding="utf-8")
    monkeypatch.setenv("MISE_DISABLE_TOOLS", ",".join(from_toml(config)["tools"]))
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(site.parent))
    return stand_in("bun", "mkdir -p node_modules")


def install(site: Path, check: bool = True) -> CompletedProcess:
    """
    Runs `site:install` through `mise run` from the root above `site`, the
    route that reads the task's sources and outputs, failing on a nonzero
    exit unless `check` is off.
    """
    return run(["mise", "run", "site:install"], check=check, cwd=site.parent)


@mark.usefixtures("bun")
def test_a_failed_install_leaves_no_marker(
    site     : Path,
    stand_in : Callable[[str, str], StandIn]
):
    """
    Asserts that an install whose Bun fails exits nonzero and touches no
    marker, so the next install runs Bun again.
    """
    stand_in("bun", "exit 1")

    assert install(site, check=False).returncode
    assert not (site / "node_modules/.installed").exists()


def test_the_install_installs_from_the_lockfile_and_prunes(bun: StandIn, site: Path):
    """
    Asserts that the install runs Bun under the frozen lockfile, then prunes
    what the lockfile no longer names, and leaves the marker its outputs
    name.
    """
    install(site)

    assert bun.commands == INSTALLED
    assert (site / "node_modules/.installed").is_file()


@mark.parametrize("name", ["bun.lock", "package.json"])
def test_the_install_reruns_once_the_manifest_or_lockfile_changes(
    bun  : StandIn,
    name : str,
    site : Path
):
    """
    Asserts that an install after `package.json` or `bun.lock` changes runs
    Bun again.
    """
    install(site)
    (site / name).write_text('{"changed": true}\n', encoding="utf-8")
    install(site)

    assert bun.commands == INSTALLED * 2


@mark.parametrize("name", ["node_modules", "node_modules/.installed"])
def test_the_install_reruns_once_node_modules_or_the_marker_is_gone(
    bun      : StandIn,
    name     : str,
    site     : Path,
    tmp_path : Path
):
    """
    Asserts that an install after `node_modules` or the marker inside it
    moves away runs Bun again.
    """
    install(site)
    (site / name).rename(tmp_path / "moved")
    install(site)

    assert bun.commands == INSTALLED * 2


def test_the_install_skips_while_the_manifest_and_lockfile_are_unchanged(
    bun  : StandIn,
    site : Path
):
    """
    Asserts that a second install over an unchanged `package.json` and
    `bun.lock` runs Bun no further.
    """
    for _ in range(2):
        install(site)

    assert bun.commands == INSTALLED
