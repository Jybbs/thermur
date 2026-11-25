"""
Pins that `site:affected` runs the whole suite where a file other than
Python was deleted since the merge base with `main`.
"""

from common.edits import git
from common.tasks import run_site
from pathlib      import Path
from pytest       import Config, mark, param

from thermur.repo.checkout import Checkout


@mark.parametrize(
    ("deleted", "moved", "passed"),
    [
        param(
            "docs/pages/index.md",
            None,
            [
                "vitest", "run", "--config",
                ".vitepress/config/vitest.config.ts", "--marker"
            ],
            id = "page"
        ),
        param(
            "docs/pages/index.md",
            "docs/pages/home.md",
            [
                "vitest", "run", "--config",
                ".vitepress/config/vitest.config.ts", "--marker"
            ],
            id = "renamed"
        ),
        param(
            "src/flock.py",
            None,
            [
                "vitest", "run", "--changed", "main",
                "--config", ".vitepress/config/vitest.config.ts", "--marker"
            ],
            id = "python"
        )
    ]
)
def test_the_affected_site_tests_run_whole_where_a_file_other_than_python_was_deleted(
    deleted      : str,
    moved        : str | None,
    passed       : list[str],
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that `site:affected` runs the whole suite where a file other
    than Python was deleted since the merge base with `main`, wherever it
    sat, a file git renames included, since Vitest selects no test for a
    deleted file that only a glob declared, and keeps `--changed main` where
    only Python was deleted.
    """
    if moved is None:
        (site.parent / deleted).unlink()
    else:
        git(Checkout(root=site.parent), "mv", deleted, moved)

    assert run_site(site, pytestconfig, "affected").stdout.splitlines() == passed
