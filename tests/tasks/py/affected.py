"""
Pins the selection `py:affected` makes for each change since `main`, and how
it fails where no `main` resolves.
"""

from collections.abc  import Callable
from common.children  import nested
from common.edits     import committed, git
from common.stand_ins import StandIn
from os               import environ, pathsep
from pathlib          import Path
from pytest           import Config, TempPathFactory, fixture, mark, param
from subprocess       import run
from sys              import executable

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
        param({".gitignore": "commit"}, "--testmon-forceselect", id="text-modified"),
        param({"docs/pages/index.md": "commit"}, "--testmon-noselect", id="page-added"),
        param({".gitignore": "moved"}, "--testmon-noselect", id="text-renamed"),
        param(
            {"tests/fixtures/flock/case.txt": "tree"},
            "--testmon-noselect",
            id = "fixture-untracked"
        ),
        param(
            {"src/thermur/flock.toml": "index"},
            "--testmon-noselect",
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
    `pyproject.toml`, or a file under `.mise` changed or a file other than
    Python was added since the merge base with `main`, a file git renames
    included, whether committed on the branch, staged, or left in the
    working tree, and with `--testmon-forceselect` where only other files
    changed. A file committed to `main` after the branch was cut counts as
    no change, and so does a file git ignores.
    """
    stand_in("pytest", 'printf "%s\\n" "$TESTMON_DATAFILE" "$@"')
    for path, where in edits.items():
        if where == "moved":
            git(branch, "mv", path, f"{path}.moved")
            continue
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


def affected(
    pytestconfig     : Config,
    repository       : Checkout,
    tmp_path_factory : TempPathFactory
) -> list[str]:
    """
    Runs `py:affected` in `repository` on the environment's own pytest
    through `nested`, with the suite's `common` helpers on the import path
    and the collection order fixed.

    Returns:
        The id of each test that passed, in the order pytest ran them.
    """
    ran = nested(
        [pytestconfig.rootpath / ".mise/tasks/py/affected", "-p", "no:randomly", "-rA"],
        tmp_path_factory,
        check = True,
        cwd   = repository.root,
        env   = {
            **environ,
            "PATH"       : f"{Path(executable).parent}{pathsep}{environ['PATH']}",
            "PYTHONPATH" : str(pytestconfig.rootpath / "tests")
        }
    ).stdout

    return [
        line.removeprefix("PASSED ") for line in ran.splitlines() if line.startswith(
            "PASSED "
        )
    ]


@fixture
def suite(
    branch           : Checkout,
    pytestconfig     : Config,
    tmp_path_factory : TempPathFactory
) -> Checkout:
    """
    Commits to the `branch` repository a `README.md`, a `guide.md`, a
    `notes.md`, a `NOTICE`, a `mark.png` holding bytes UTF-8 cannot decode,
    the suite's own conftest, and a suite of ten tests:

    - One opening the README
    - One opening the copy of it `checkout` makes, which the hook maps back onto it
    - One opening the copy of it `shared` holds, which the hook maps back onto it
    - One opening the README while the hook is paused
    - One opening the README before it requests a fixture by name
    - Two requesting a session fixture that opens the guide once
    - One opening the `NOTICE`
    - One opening the mark
    - One opening nothing

    It then moves `main` onto that commit, so no file the suite adds counts
    as added since `main`, and runs `py:affected` once so pytest-testmon
    records what each test read.

    Returns:
        The `Checkout` rooted at the repository.
    """
    for path, text in {
        "NOTICE"              : "A notice.\n",
        "README.md"           : "A flock.\n",
        "guide.md"            : "A guide.\n",
        "notes.md"            : "Unread.\n",
        "tests/conftest.py"   : (pytestconfig.rootpath / "tests/conftest.py").read_text(
            encoding = "utf-8"
        ),
        "tests/test_flock.py" : (
            "from pathlib import Path\n"
            "from pytest import fixture\n\n"
            "@fixture(scope='session')\n"
            "def guide():\n"
            "    return Path('guide.md').read_text()\n\n"
            "def test_reads():\n"
            "    assert Path('README.md').read_text()\n\n"
            "def test_copies(checkout):\n"
            "    assert (checkout.root / 'README.md').read_text()\n\n"
            "def test_shares(checked):\n"
            "    assert (checked.root / 'README.md').read_text()\n\n"
            "def test_guides_first(guide):\n"
            "    assert guide\n\n"
            "def test_guides_second(guide):\n"
            "    assert guide\n\n"
            "def test_reads_paused(monkeypatch, opens):\n"
            "    monkeypatch.setattr(opens, 'paused', True)\n"
            "    assert Path('README.md').read_text()\n\n"
            "def test_reads_notice():\n"
            "    assert Path('NOTICE').read_text()\n\n"
            "def test_reads_mark():\n"
            "    assert Path('mark.png').read_bytes()\n\n"
            "def test_reads_then_requests(request):\n"
            "    assert Path('README.md').read_text()\n"
            "    assert request.getfixturevalue('tmp_path')\n\n"
            "def test_reads_nothing():\n"
            "    assert True\n"
        )
    }.items():
        (branch.root / path).parent.mkdir(exist_ok=True, parents=True)
        (branch.root / path).write_text(text, encoding="utf-8")
    (branch.root / "mark.png").write_bytes(b"\x89PNG\xff")
    git(branch, "add", "--all")
    git(branch, "commit", "--message=Suite", "--quiet")
    git(branch, "update-ref", "refs/heads/main", "HEAD")
    affected(pytestconfig, branch, tmp_path_factory)

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
            [
                "tests/test_flock.py::test_copies", "tests/test_flock.py::test_reads",
                "tests/test_flock.py::test_reads_then_requests",
                "tests/test_flock.py::test_shares"
            ],
            id = "read"
        ),
        param(
            "guide.md",
            [
                "tests/test_flock.py::test_guides_first",
                "tests/test_flock.py::test_guides_second"
            ],
            id = "fixture"
        ),
        param("notes.md", [], id="unread"),
        param("mark.png", [], id="unfingerprinted"),
        param("NOTICE", [], id="extensionless"),
        param(
            "uv.lock",
            [
                "tests/test_flock.py::test_copies",
                "tests/test_flock.py::test_guides_first",
                "tests/test_flock.py::test_guides_second",
                "tests/test_flock.py::test_reads",
                "tests/test_flock.py::test_reads_mark",
                "tests/test_flock.py::test_reads_nothing",
                "tests/test_flock.py::test_reads_notice",
                "tests/test_flock.py::test_reads_paused",
                "tests/test_flock.py::test_reads_then_requests",
                "tests/test_flock.py::test_shares"
            ],
            id = "lockfile"
        )
    ]
)
def test_the_affected_tests_are_those_opening_a_changed_file(
    passed           : list[str],
    path             : str,
    pytestconfig     : Config,
    suite            : Checkout,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that `py:affected` runs only the tests that opened a changed
    file other than Python, each test opening a copy the `Opened` hook maps
    back onto that file and each test requesting a session fixture that
    opened it included, and the test opening it while the hook is paused
    left out. It runs none for a file no test opened or one pytest-testmon
    cannot fingerprint as text, whose reader it still records, and runs
    every test where `uv.lock` changed.
    """
    (suite.root / path).write_text("Changed.\n", encoding="utf-8")

    assert sorted(affected(pytestconfig, suite, tmp_path_factory)) == passed
