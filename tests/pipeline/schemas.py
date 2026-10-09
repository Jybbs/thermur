"""
Pins the pipeline's records, meaning what a `Run` reads from its clone and
what it refuses, which folders `Subpackage` lists and the steps module each
holds, and which edits change a subpackage's digest.
"""

from pathlib  import Path
from platform import platform
from pydantic import ValidationError
from pytest   import Config, MonkeyPatch, mark, param, raises

from thermur.pipeline.schemas  import Run, Subpackage
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout

MODULE = '''"""
Holds the stand-in step.
"""


def doubled(value: float) -> float:
    """
    Doubles `value`.
    """
    return value * 2  # The stand-in factor.
'''


@mark.parametrize(
    ("path", "text", "changes"),
    [
        param("module.py", MODULE.replace("* 2", "* 3"), True, id="statement"),
        param("module.py", MODULE.replace("Doubles", "Twice"), False, id="docstring"),
        param(
            "module.py",
            MODULE.replace("stand-in factor", "factor"),
            False,
            id = "comment"
        ),
        param("norms.toml", "factor = 3\n", True, id="data-file"),
        param("added.py", "LIMIT = 1\n", True, id="new-module"),
        param("__pycache__/module.cpython-314.pyc", "\x00", False, id="bytecode")
    ]
)
def test_a_digest_changes_wherever_the_code_does(
    changes  : bool,
    path     : str,
    text     : str,
    tmp_path : Path
):
    """
    Asserts that a subpackage's digest changes with a statement, a data
    file, or a new module, and holds through an edit to a function's
    docstring or a comment and through the bytecode Python writes beside
    a module.
    """
    original = tmp_path / "original"
    original.mkdir()
    (original / "module.py").write_text(MODULE)
    (original / "norms.toml").write_text("factor = 2\n")
    edited = original.copy(tmp_path / "edited")
    (edited / path).parent.mkdir(exist_ok=True)
    (edited / path).write_text(text)

    before, after = (
        Subpackage(folder=folder).digest("lock") for folder in (original, edited)
    )

    assert (after != before) is changes


def test_a_digest_changes_with_the_lockfile(tmp_path: Path):
    """
    Asserts that a subpackage whose files stay unchanged digests apart under
    two digests of `uv.lock`, so a library upgrade computes its steps again.
    """
    (tmp_path / "module.py").write_text(MODULE)
    subpackage = Subpackage(folder=tmp_path)

    assert subpackage.digest("one") != subpackage.digest("two")


@mark.parametrize(
    "commit",
    [
        param("fatal: not a git repository", id="an-error-message"),
        param("", id="nothing")
    ]
)
def test_a_run_refuses_a_commit_holding_no_hash(commit: str):
    """
    Asserts that a commit holding anything but hexadecimal digits is
    refused, so a record never carries what git printed in place of a hash.
    """
    with raises(ValidationError):
        Run(commit=commit, lockfile="0" * 64, settings=Settings())


def test_a_run_reads_its_commit_and_lockfile_from_its_clone(clone: Checkout):
    """
    Asserts that a run pairs its settings with the commit its clone has
    checked out, the digest of the clone's `uv.lock`, and the running
    platform.
    """
    assert Run.from_checkout(clone, Settings()) == Run(
        commit   = clone.commit,
        lockfile = clone.lockfile,
        platform = platform(),
        settings = Settings()
    )


def test_listed_holds_every_folder_of_the_package(pytestconfig: Config):
    """
    Asserts that the subpackages listed are every folder under
    `src/thermur`, in name order.
    """
    assert [subpackage.folder for subpackage in Subpackage.listed()] == sorted(
        (pytestconfig.rootpath / "src/thermur").glob("*/")
    )


def test_listed_leaves_out_a_file_at_the_package_s_root(
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that a file beside the subpackages, such as the `.DS_Store`
    Finder writes into a folder it opens, lists as no subpackage.
    """
    for name in ("alpha", "beta"):
        (tmp_path / name).mkdir()
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    monkeypatch.setattr("thermur.pipeline.schemas.files", lambda _: tmp_path)

    assert [subpackage.name for subpackage in Subpackage.listed()] == ["alpha", "beta"]


@mark.parametrize(
    ("files", "steps"),
    [
        param(["steps.py"], "thermur.stand.steps", id="steps"),
        param(["schemas.py"], None, id="no-steps")
    ]
)
def test_steps_names_the_module_a_subpackage_holds(
    files    : list[str],
    steps    : str | None,
    tmp_path : Path
):
    """
    Asserts that a subpackage holding a `steps.py` names its dotted module,
    and that one holding none names nothing.
    """
    folder = tmp_path / "stand"
    folder.mkdir()
    for name in files:
        (folder / name).write_text("")

    assert Subpackage(folder=folder).steps == steps
