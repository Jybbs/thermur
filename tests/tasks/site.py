"""
Pins the arguments each `site` task hands the tool it runs, meaning the copy
under `docs/node_modules/.bin` each runs, every argument a reader adds, and
the annotations the lint prints under GitHub Actions, and the environment
every `bun` command in the clone runs under.
"""

from common.tasks import run_site
from pathlib      import Path
from pytest       import Config, MonkeyPatch, mark, param
from subprocess   import run


@mark.parametrize(
    ("task", "actions", "passed"),
    [
        param(
            "lint",
            "true",
            [
                "oxlint", "--config", ".vitepress/config/.oxlintrc.json", "--format",
                "github", "--marker", ".vitepress", "kit"
            ],
            id = "lint-actions"
        ),
        param(
            "lint",
            None,
            [
                "oxlint", "--config", ".vitepress/config/.oxlintrc.json", "--format",
                "default", "--marker", ".vitepress", "kit"
            ],
            id = "lint-unset"
        ),
        param(
            "cruft",
            "true",
            [
                "knip", "--reporter", "github-actions", "--marker", "knip",
                "--cycles", "--reporter", "github-actions", "--marker"
            ],
            id = "cruft-actions"
        ),
        param(
            "cruft",
            "",
            [
                "knip", "--reporter", "symbols", "--marker", "knip",
                "--cycles", "--reporter", "symbols", "--marker"
            ],
            id = "cruft-empty"
        ),
        param("press", None, ["vitepress", "build", "--marker"], id="press"),
        param("serve", None, ["vitepress", "dev", "--marker"], id="serve"),
        param(
            "test",
            None,
            [
                "vitest", "run", "--config",
                ".vitepress/config/vitest.config.ts", "--marker"
            ],
            id = "test"
        ),
        param(
            "affected",
            None,
            [
                "vitest", "run", "--changed", "main",
                "--config", ".vitepress/config/vitest.config.ts", "--marker"
            ],
            id = "affected"
        ),
        param(
            "coverage",
            None,
            [
                "vitest", "run", "--config",
                ".vitepress/config/vitest.config.ts", "--coverage", "--marker"
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

    assert run_site(site, pytestconfig, task).stdout.splitlines() == passed


def test_a_bun_command_runs_no_install_script_and_caches_under_the_clone(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config
):
    """
    Asserts that the environment mise sets for every command in the clone,
    a `bun` typed by hand included, skips each package's install scripts
    and puts Bun's install cache under `.cache/bun`, clearing both variables
    first so neither reaches the command from the shell running the suite.
    """
    for name in ("BUN_INSTALL_CACHE_DIR", "NPM_CONFIG_IGNORE_SCRIPTS"):
        monkeypatch.delenv(name, raising=False)
    printed = run(
        [
            "mise", "x",
            "--", "sh",
            "-c", 'printf "%s\\n" "$NPM_CONFIG_IGNORE_SCRIPTS" "$BUN_INSTALL_CACHE_DIR"'
        ],
        capture_output = True,
        check          = True,
        cwd            = pytestconfig.rootpath / "docs",
        text           = True
    )

    assert printed.stdout.splitlines() == [
        "true",
        str(pytestconfig.rootpath / ".cache/bun")
    ]
