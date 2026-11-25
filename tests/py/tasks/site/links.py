"""
Pins that the link check reads the built site, the README, and the
contributor guide offline.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import run_site
from pathlib          import Path
from pytest           import Config


def test_the_link_check_reads_the_built_site_and_the_guides_offline(
    pytestconfig : Config,
    site         : Path,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that the link check runs lychee offline over the built site,
    the README, and the contributor guide, resolving an absolute link under
    the built site and mapping each address under the homepage Node prints
    onto it.
    """
    dist = site / ".vitepress/dist"
    stand_in("lychee", 'printf "%s\\n" "$@"')
    stand_in("node", "echo https://example.org/")

    assert run_site(site, pytestconfig, "links").stdout.splitlines() == [
        "--fallback-extensions", "html", "--include-fragments", "--index-files",
        "index.html", "--no-progress", "--offline", "--remap",
        f"^https://example.org/ file://{dist}/",
        "--root-dir",
        str(dist),
        "--marker",
        str(dist),
        "../.github/README.md", "../.github/CONTRIBUTING.md"
    ]
