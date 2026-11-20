"""
Pins the arguments each `py` task hands the formatter or pytest, meaning the
folders the formatter reads, the reports coverage writes, every argument a
reader adds, the annotations `py:check` prints under GitHub Actions, and the
selection `py:affected` makes for each change since `main`.
"""

from collections.abc  import Callable
from common.edits     import committed, git
from common.stand_ins import StandIn
from os               import environ, pathsep
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture, mark, param
from subprocess       import run
from sys              import executable

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
    committed(repository)
    git(repository, "switch", "--create", "change", "--quiet")

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
        param({"docs/pages/index.md": "commit"}, "--testmon-forceselect", id="page"),
        param(
            {"tests/fixtures/flock/case.txt": "tree"},
            "--testmon-forceselect",
            id = "fixture-untracked"
        ),
        param(
            {"src/thermur/flock.toml": "index"},
            "--testmon-forceselect",
            id = "source-data-staged"
        ),
        param({"uv.lock": "main"}, "--testmon-forceselect", id="main-only"),
        param({".cache/testmon/data": "tree"}, "--testmon-forceselect", id="ignored"),
        param({"uv.lock": "commit"}, "--testmon-noselect", id="lockfile"),
        param(
            {"pyproject.toml": "tree", "src/thermur/flock.py": "tree"},
            "--testmon-noselect",
            id = "manifest-untracked"
        ),
        param({".mise/tasks/py/test": "index"}, "--testmon-noselect", id="task"),
        param(
            {".mise/config.toml": "commit", "docs/pages/index.md": "commit"},
            "--testmon-noselect",
            id = "mise-beside-page"
        )
    ]
)
def test_the_affected_tests_select_unless_a_shared_input_changed(
    branch       : Checkout,
    edits        : dict[str, str],
    pytestconfig : Config,
    selection    : str,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that `py:affected` runs pytest under the testmon data file
    in `.cache/testmon` with `--testmon-noselect` where `uv.lock`,
    `pyproject.toml`, or a file under `.mise` changed since the merge base
    with `main`, whether committed on the branch, staged, or left in the
    working tree, and with `--testmon-forceselect` where only other files
    changed. A file committed to `main` after the branch was cut counts as
    no change, and so does a file git ignores.
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


def affected(pytestconfig: Config, repository: Checkout) -> list[str]:
    """
    Runs `py:affected` in `repository` on the environment's own pytest, with
    the suite's `common` helpers on the import path and the collection order
    fixed.

    Returns:
        The id of each test that passed, in the order pytest ran them.
    """
    ran = run(
        [pytestconfig.rootpath / ".mise/tasks/py/affected", "-p", "no:randomly", "-rA"],
        capture_output = True,
        check          = True,
        cwd            = repository.root,
        env            = {
            **environ,
            "PATH"       : f"{Path(executable).parent}{pathsep}{environ['PATH']}",
            "PYTHONPATH" : str(pytestconfig.rootpath / "tests")
        },
        text = True
    ).stdout

    return [
        line.removeprefix("PASSED ") for line in ran.splitlines() if line.startswith(
            "PASSED "
        )
    ]


@fixture
def suite(branch: Checkout, pytestconfig: Config) -> Checkout:
    """
    Commits to the `branch` repository a `README.md`, a `notes.md`, a
    `NOTICE`, a `mark.png` holding bytes UTF-8 cannot decode, a conftest
    holding the `opens` and `recorded` fixtures the suite's own conftest
    holds, and a suite of six tests:

    - One opening the README
    - One opening a copy of it under `tmp_path` that the hook maps back onto it
    - One opening the README while the hook is paused
    - One opening the `NOTICE`
    - One opening the mark
    - One opening nothing

    It then runs `py:affected` once so pytest-testmon records what each
    test read.

    Returns:
        The `Checkout` rooted at the repository.
    """
    for path, text in {
        "NOTICE"            : "A notice.\n",
        "README.md"         : "A flock.\n",
        "notes.md"          : "Unread.\n",
        "tests/conftest.py" : (
            "from common.opened import Opened\n"
            "from pytest import fixture\n"
            "from thermur.repo.checkout import Checkout\n\n"
            "@fixture(autouse=True, scope='session')\n"
            "def opens(pytestconfig):\n"
            "    root = pytestconfig.rootpath\n"
            "    return Opened.added(pytestconfig, Checkout(root=root).tracked())\n\n"
            "@fixture(autouse=True)\n"
            "def recorded(opens):\n"
            "    yield\n"
            "    opens.record()\n"
        ),
        "tests/test_flock.py": (
            "from pathlib import Path\n\n"
            "def test_reads():\n"
            "    assert Path('README.md').read_text()\n\n"
            "def test_copies(monkeypatch, opens, tmp_path):\n"
            "    (tmp_path / 'README.md').write_bytes(Path('README.md').read_bytes())\n"
            "    monkeypatch.setitem(opens.copies, tmp_path, Path.cwd())\n"
            "    assert (tmp_path / 'README.md').read_text()\n\n"
            "def test_reads_paused(monkeypatch, opens):\n"
            "    monkeypatch.setattr(opens, 'paused', True)\n"
            "    assert Path('README.md').read_text()\n\n"
            "def test_reads_notice():\n"
            "    assert Path('NOTICE').read_text()\n\n"
            "def test_reads_mark():\n"
            "    assert Path('mark.png').read_bytes()\n\n"
            "def test_reads_nothing():\n"
            "    assert True\n"
        )
    }.items():
        (branch.root / path).parent.mkdir(exist_ok=True, parents=True)
        (branch.root / path).write_text(text, encoding="utf-8")
    (branch.root / "mark.png").write_bytes(b"\x89PNG\xff")
    git(branch, "add", "--all")
    git(branch, "commit", "--message=Suite", "--quiet")
    affected(pytestconfig, branch)

    return branch


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


@mark.parametrize(
    ("path", "passed"),
    [
        param(
            "README.md",
            ["tests/test_flock.py::test_copies", "tests/test_flock.py::test_reads"],
            id = "read"
        ),
        param("notes.md", [], id="unread"),
        param("mark.png", [], id="unfingerprinted"),
        param("NOTICE", [], id="extensionless"),
        param(
            "uv.lock",
            [
                "tests/test_flock.py::test_copies", "tests/test_flock.py::test_reads",
                "tests/test_flock.py::test_reads_mark",
                "tests/test_flock.py::test_reads_nothing",
                "tests/test_flock.py::test_reads_notice",
                "tests/test_flock.py::test_reads_paused"
            ],
            id = "lockfile"
        )
    ]
)
def test_the_affected_tests_are_those_opening_a_changed_file(
    passed       : list[str],
    path         : str,
    pytestconfig : Config,
    suite        : Checkout
):
    """
    Asserts that `py:affected` runs only the tests that opened a changed
    file other than Python, the test opening a copy the `Opened` hook
    maps back onto that file included and the test opening it while the
    hook is paused left out. It runs none for a file no test opened or
    one pytest-testmon cannot fingerprint as text, whose reader it still
    records, and runs every test where `uv.lock` changed.
    """
    (suite.root / path).write_text("Changed.\n", encoding="utf-8")

    assert sorted(affected(pytestconfig, suite)) == passed
