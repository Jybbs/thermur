"""
Pins that the install runs Bun under the frozen lockfile, then prunes, then
links `tests/node_modules` to the site's packages.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import run_site
from pathlib          import Path
from pytest           import Config


def test_a_second_install_leaves_the_link_as_it_stands(
    pytestconfig : Config,
    site         : Path,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that an install run over an earlier one replaces the link
    at `tests/node_modules` rather than writing a second link inside
    `docs/node_modules`, which every `site` task installs through.
    """
    stand_in("bun", "exit 0")
    run_site(site, pytestconfig, "install")
    run_site(site, pytestconfig, "install")

    assert [path.name for path in (site / "node_modules").iterdir()] == [".bin"]


def test_the_install_installs_from_the_lockfile_and_prunes(
    pytestconfig : Config,
    site         : Path,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that the install runs Bun under the frozen lockfile, then prunes
    what the lockfile no longer names, then links `tests/node_modules`
    to `docs/node_modules` by a relative path, so a suite under `tests/`
    resolves the site's packages in any clone.
    """
    bun = stand_in("bun", "exit 0")
    run_site(site, pytestconfig, "install")

    assert bun.commands == ["bun install --frozen-lockfile", "bun prune"]
    assert (site.parent / "tests/node_modules").readlink() == Path(
        "../docs/node_modules"
    )
