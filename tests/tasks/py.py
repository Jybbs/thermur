"""
Pins the arguments each `py` task hands the formatter or pytest, meaning the
folders the formatter reads, the reports coverage writes, every argument a
reader adds, the annotations `py:check` prints under GitHub Actions, and the
selection `py:affected` makes for each change since `main`.
"""

from collections.abc  import Callable
from common.edits     import git
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture, mark, param
from subprocess       import run

from thermur.repo.checkout import Checkout

FOLDERS = [".mise/tasks", "src", "tests"]


@mark.parametrize(
    ("task", "actions", "passed"),
    [
        param(
            "check",
            "true",
            ["check", "--output-format", "github", "--marker", *FOLDERS],
            id = "check-actions"
        ),
        param(
            "check",
            "",
            ["check", "--output-format", "text", "--marker", *FOLDERS],
            id = "check-empty"
        ),
        param(
            "check",
            None,
            ["check", "--output-format", "text", "--marker", *FOLDERS],
            id = "check-unset"
        ),
        param("format", None, ["format", "--marker", *FOLDERS], id="format"),
        param("test", None, ["--marker"], id="test"),
        param(
            "coverage",
            None,
            ["--cov", "--cov-report", "html", "--cov-report", "term", "--marker"],
            id = "coverage"
        )
    ]
)
def test_each_task_hands_its_program_every_argument(
    actions      : str | None,
    monkeypatch  : MonkeyPatch,
    passed       : list[str],
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn],
    task         : str
):
    """
    Asserts that each `py` task runs its program on the arguments it
    fixes followed by every argument a reader adds, the formatter over
    `.mise/tasks`, `src`, and `tests` and coverage writing its HTML and
    terminal reports, with `py:check` passing `--output-format github` where
    `GITHUB_ACTIONS` is set and `--output-format text` where it is empty
    or unset.
    """
    root = pytestconfig.rootpath
    for program in ("prose", "pytest"):
        stand_in(program, 'printf "%s\\n" "$@"')

    if actions is not None:
        monkeypatch.setenv("GITHUB_ACTIONS", actions)

    printed = run(
        [root / ".mise/tasks/py" / task, "--marker"],
        capture_output = True,
        check          = True,
        cwd            = root,
        text           = True
    ).stdout.splitlines()

    assert printed == passed


@fixture
def branch(tmp_path: Path) -> Checkout:
    """
    Commits `src/thermur/flock.py`, an empty `uv.lock`, and a `.gitignore`
    ignoring `.cache/` to `main` in a git repository under `tmp_path`,
    then switches to a branch cut from that commit, where a case makes its
    change.

    Returns:
        The `Checkout` rooted at the repository.
    """
    repository = Checkout(root=tmp_path)
    (tmp_path / "src/thermur").mkdir(parents=True)
    for path, text in {
        ".gitignore"           : "/.cache/\n",
        "src/thermur/flock.py" : "",
        "uv.lock"              : ""
    }.items():
        (tmp_path / path).write_text(text, encoding="utf-8")
    for command in (
        ["init", "--initial-branch=main", "--quiet"],
        ["add", "--all"],
        ["commit", "--message=Start", "--quiet"],
        ["switch", "--create", "change", "--quiet"]
    ):
        git(repository, *command)

    return repository


@mark.parametrize(
    ("edits", "selection"),
    [
        param({}, "--testmon-forceselect", id="none"),
        param(
            {"src/thermur/flock.py": "tree"},
            "--testmon-forceselect",
            id = "source-modified"
        ),
        param({"tests/flock.py": "tree"}, "--testmon-forceselect", id="test-untracked"),
        param(
            {"src/thermur/flock.py": "commit", "tests/flock.py": "commit"},
            "--testmon-forceselect",
            id = "python-committed"
        ),
        param({"docs/pages/index.md": "main"}, "--testmon-forceselect", id="main-only"),
        param({".cache/testmon/data": "tree"}, "--testmon-forceselect", id="ignored"),
        param(
            {".mise/tasks/docs/render.py": "commit"},
            "--testmon-noselect",
            id = "task-python"
        ),
        param(
            {"tests/fixtures/flock/case.txt": "tree"},
            "--testmon-noselect",
            id = "fixture-untracked"
        ),
        param({"docs/pages/index.md": "commit"}, "--testmon-noselect", id="page"),
        param({".mise/tasks/py/test": "commit"}, "--testmon-noselect", id="task"),
        param(
            {"src/thermur/flock.toml": "tree"},
            "--testmon-noselect",
            id = "source-data-untracked"
        ),
        param(
            {"tests/fixtures/flock/case.txt": "index"},
            "--testmon-noselect",
            id = "fixture-staged"
        ),
        param(
            {"src/thermur/flock.py": "tree", "uv.lock": "tree"},
            "--testmon-noselect",
            id = "lockfile-beside-source"
        )
    ]
)
def test_the_affected_tests_select_unless_a_change_lies_outside_the_python(
    branch       : Checkout,
    edits        : dict[str, str],
    pytestconfig : Config,
    selection    : str,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that `py:affected` runs pytest under the testmon data file in
    `.cache/testmon` with `--testmon-forceselect` where every file changed
    since the merge base with `main` is Python under `src` or `tests`,
    and with `--testmon-noselect` where any other file changed, whether
    committed on the branch, staged, or left in the working tree. A file
    committed to `main` after the branch was cut counts as no change, and so
    does a file git ignores.
    """
    stand_in("pytest", 'printf "%s\\n" "$TESTMON_DATAFILE" "$@"')
    for path, where in edits.items():
        if where == "main":
            git(branch, "switch", "--quiet", "main")
        (branch.root / path).parent.mkdir(exist_ok=True, parents=True)
        (branch.root / path).write_text("changed", encoding="utf-8")
        if where != "tree":
            git(branch, "add", "--all")
        if where in {"commit", "main"}:
            git(branch, "commit", "--message=Change", "--quiet")
        if where == "main":
            git(branch, "switch", "--quiet", "change")

    printed = run(
        [pytestconfig.rootpath / ".mise/tasks/py/affected", "--marker"],
        capture_output = True,
        check          = True,
        cwd            = branch.root,
        text           = True
    ).stdout.splitlines()

    assert printed == [".cache/testmon/data", selection, "--marker"]
    assert (branch.root / ".cache/testmon").is_dir()


def test_the_affected_tests_fail_where_no_main_resolves(
    branch       : Checkout,
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that `py:affected` exits nonzero without running pytest where
    the repository holds no `main` to take the merge base with.
    """
    stand_in("pytest", "echo ran")
    git(branch, "branch", "--move", "main", "trunk")
    ran = run(
        [pytestconfig.rootpath / ".mise/tasks/py/affected"],
        capture_output = True,
        cwd            = branch.root,
        text           = True
    )

    assert (ran.returncode != 0, ran.stdout) == (True, "")
