"""
Pins the selection `py:affected` makes for each change since `main`, and how
it fails where no `main` resolves.
"""

from collections.abc  import Callable
from common.edits     import git
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import Config, fixture, mark, param
from subprocess       import run

from thermur.repo.checkout import Checkout


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
