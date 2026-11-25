"""
Pins that the install runs Bun under the frozen lockfile and then prunes,
skipping both while `docs/package.json` and `docs/bun.lock` are unchanged
and `docs/node_modules` exists, and that no other mise command installs.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import run_site
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture, mark
from subprocess       import run
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
    Copies `.mise/config.toml` above `site` and trusts it, turning off each
    tool it pins, which mise would otherwise install into the empty `HOME`
    before running the provider. It then writes a `package.json` and a
    `bun.lock` into `site` and puts a stand-in `bun` on the path that writes
    `node_modules` where it runs.
    """
    (site.parent / ".mise").mkdir()
    config = (pytestconfig.rootpath / ".mise/config.toml").copy(
        site.parent / ".mise/config.toml"
    )
    for name in ("package.json", "bun.lock"):
        (site / name).write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv(
        "MISE_DISABLE_TOOLS",
        ",".join(from_toml(config.read_text(encoding="utf-8"))["tools"])
    )
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(site.parent))
    return stand_in("bun", "mkdir -p node_modules")


def test_another_mise_command_installs_nothing(bun: StandIn, site: Path):
    """
    Asserts that `mise exec` leaves the install to the task, since the
    provider stays off `auto`.
    """
    run(["mise", "exec", "--", "true"], check=True, cwd=site)

    assert bun.commands == []


def test_the_install_installs_from_the_lockfile_and_prunes(
    bun          : StandIn,
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that the install runs Bun under the frozen lockfile, then prunes
    what the lockfile no longer names.
    """
    run_site(site, pytestconfig, "install")

    assert bun.commands == INSTALLED


@mark.parametrize("name", ["bun.lock", "package.json"])
def test_the_install_reruns_once_the_manifest_or_lockfile_changes(
    bun          : StandIn,
    name         : str,
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that an install after `package.json` or `bun.lock` changes runs
    Bun again.
    """
    run_site(site, pytestconfig, "install")
    (site / name).write_text('{"changed": true}\n', encoding="utf-8")
    run_site(site, pytestconfig, "install")

    assert bun.commands == INSTALLED * 2


def test_the_install_reruns_once_node_modules_is_gone(
    bun          : StandIn,
    pytestconfig : Config,
    site         : Path,
    tmp_path     : Path
):
    """
    Asserts that an install after `node_modules` moves away runs Bun again.
    """
    run_site(site, pytestconfig, "install")
    (site / "node_modules").rename(tmp_path / "moved")
    run_site(site, pytestconfig, "install")

    assert bun.commands == INSTALLED * 2


def test_the_install_skips_while_the_manifest_and_lockfile_are_unchanged(
    bun          : StandIn,
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that a second install over an unchanged `package.json` and
    `bun.lock` runs Bun no further.
    """
    for _ in range(2):
        run_site(site, pytestconfig, "install")

    assert bun.commands == INSTALLED
