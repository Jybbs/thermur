"""
Pins the arguments each `site` task hands the tool it runs, meaning the
copy under `docs/node_modules/.bin` each runs, every argument a reader adds,
the annotations the lint and the cruft report print under GitHub Actions,
the single-file component the type check refuses, and the built site, the
README, and the contributor guide the link check reads offline.
"""

from collections.abc  import Callable
from common.edits     import executable
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture, mark, param
from subprocess       import CompletedProcess, run

ECHO = '#!/bin/sh\nprintf "%s\\n" "$(basename "$0")" "$@"\n'


@fixture
def site(tmp_path: Path) -> Path:
    """
    Writes a `docs/` folder holding an empty `.vitepress` and a
    `node_modules/.bin` holding a stand-in for each tool a `site` task runs,
    each printing its own name and every argument it receives.

    Returns:
        The `docs/` folder, which every `site` task runs from.
    """
    (tmp_path / "docs/.vitepress").mkdir(parents=True)
    for tool in ("knip", "oxlint", "tsc", "vitepress", "vitest"):
        executable(tmp_path / "docs/node_modules/.bin" / tool, ECHO)

    return tmp_path / "docs"


def ran(cwd: Path, pytestconfig: Config, task: str) -> CompletedProcess:
    """
    Runs the `site` task named through its own shebang from `cwd`, with
    `--marker` as an argument a reader adds, capturing what it prints.
    """
    return run(
        [pytestconfig.rootpath / ".mise/tasks/site" / task, "--marker"],
        capture_output = True,
        cwd            = cwd,
        text           = True
    )


@mark.parametrize(
    ("task", "actions", "passed"),
    [
        param(
            "lint",
            "true",
            [
                "oxlint", "--config", ".vitepress/.oxlintrc.json", "--format",
                "github", "--marker", ".vitepress"
            ],
            id = "lint-actions"
        ),
        param(
            "lint",
            None,
            [
                "oxlint", "--config", ".vitepress/.oxlintrc.json", "--format",
                "default", "--marker", ".vitepress"
            ],
            id = "lint-unset"
        ),
        param(
            "cruft",
            "true",
            [
                "knip", "--config", ".vitepress/knip.ts", "--reporter",
                "github-actions", "--marker", "knip",
                "--config", ".vitepress/knip.ts", "--cycles",
                "--reporter", "github-actions", "--marker"
            ],
            id = "cruft-actions"
        ),
        param(
            "cruft",
            "",
            [
                "knip", "--config", ".vitepress/knip.ts", "--reporter", "symbols",
                "--marker", "knip", "--config", ".vitepress/knip.ts",
                "--cycles", "--reporter", "symbols", "--marker"
            ],
            id = "cruft-empty"
        ),
        param("press", None, ["vitepress", "build", "--marker"], id="press"),
        param("serve", None, ["vitepress", "dev", "--marker"], id="serve"),
        param(
            "test",
            None,
            ["vitest", "run", "--config", ".vitepress/vitest.config.ts", "--marker"],
            id = "test"
        ),
        param(
            "coverage",
            None,
            [
                "vitest", "run", "--config",
                ".vitepress/vitest.config.ts", "--coverage", "--marker"
            ],
            id = "coverage"
        ),
        param(
            "typecheck",
            None,
            ["tsc", "--project", ".vitepress/tsconfig.json", "--marker"],
            id = "typecheck"
        )
    ]
)
def test_each_task_runs_its_tool_from_the_site_s_own_packages(
    actions      : str | None,
    monkeypatch  : MonkeyPatch,
    passed       : list[str],
    pytestconfig : Config,
    site         : Path,
    task         : str
):
    """
    Asserts that each `site` task runs the copy of its tool under
    `docs/node_modules/.bin` on the arguments it fixes followed by every
    argument a reader adds, the cruft report running knip a second time over
    import cycles alone. The lint prints annotations and the cruft report
    takes the GitHub Actions reporter where `GITHUB_ACTIONS` is set.
    """
    if actions is not None:
        monkeypatch.setenv("GITHUB_ACTIONS", actions)

    assert ran(site, pytestconfig, task).stdout.splitlines() == passed


def test_the_install_installs_from_the_lockfile_and_prunes(
    pytestconfig : Config,
    site         : Path,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that the install runs Bun under the frozen lockfile, then prunes
    what the lockfile no longer names.
    """
    bun = stand_in("bun", "exit 0")
    ran(site, pytestconfig, "install")

    assert bun.commands == ["bun install --frozen-lockfile", "bun prune"]


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

    assert ran(site, pytestconfig, "links").stdout.splitlines() == [
        "--fallback-extensions", "html", "--include-fragments", "--index-files",
        "index.html", "--no-progress", "--offline", "--remap",
        f"^https://example.org/ file://{dist}/",
        "--root-dir",
        str(dist),
        "--marker",
        str(dist),
        "../README.md", "../.github/CONTRIBUTING.md"
    ]


def test_the_type_check_refuses_a_single_file_component(
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that the type check fails before `tsc` runs, naming each
    single-file component under `.vitepress`, since `tsc` reads none.
    """
    (site / ".vitepress/Card.vue").write_text("", encoding="utf-8")
    checked = ran(site, pytestconfig, "typecheck")

    assert (checked.returncode, checked.stdout) == (1, "")
    assert checked.stderr.rstrip().endswith(".vitepress/Card.vue")
