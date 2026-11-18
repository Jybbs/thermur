"""
Pins the pipeline's records, meaning what a `Run` reads from its clone and
what it refuses, which folders `Subpackage` lists and the steps module each
holds, and which edits change a subpackage's digest, its own or those of the
modules it imports.
"""

from common.edits import append
from pathlib      import Path
from platform     import platform
from pydantic     import ValidationError
from pytest       import Config, MonkeyPatch, mark, param, raises

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
        param("uv.lock", "version = 1\n", True, id="file-named-uv-lock"),
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


@mark.parametrize(
    ("path", "text", "changes"),
    [
        param("imported/module.py", "LIMIT = 1\n", True, id="imported-module"),
        param("deeper/module.py", "LIMIT = 1\n", True, id="module-imported-in-turn"),
        param("imported/norms.toml", "factor = 3\n", True, id="imported-data-file"),
        param(
            "deeper/norms.toml",
            "factor = 3\n",
            True,
            id = "data-file-imported-in-turn"
        ),
        param("unimported/module.py", "LIMIT = 1\n", False, id="unimported-module")
    ]
)
def test_a_digest_follows_the_package_imports_of_its_modules(
    changes  : bool,
    path     : str,
    text     : str,
    tmp_path : Path
):
    """
    Asserts that a subpackage's digest changes with a module of another
    subpackage one of its modules imports, a module that module imports
    in turn, and a data file beside either, and holds through an edit to a
    module none of them imports.
    """
    package = tmp_path / "original/thermur"
    for name in ("deeper", "imported", "reader", "unimported"):
        (package / name).mkdir(parents=True)
        (package / name / "module.py").write_text(MODULE)
    (package / "imported/norms.toml").write_text("factor = 2\n")
    (package / "imported/module.py").write_text(
        f"from thermur.deeper import module\n{MODULE}"
    )
    (package / "reader/module.py").write_text(
        "import os\nimport thermur.imported.module\n"
    )
    edited = package.parent.copy(tmp_path / "edited") / "thermur"
    append(Checkout(root=edited), path, text)

    before, after = (
        Subpackage(folder=root / "reader").digest("lock") for root in (package, edited)
    )

    assert (after != before) is changes


def test_a_digest_changes_with_the_lock_digest(tmp_path: Path):
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
        Run(commit=commit, lock_digest="0" * 64, settings=Settings())


def test_a_run_reads_its_commit_and_lock_digest_from_its_clone(clone: Checkout):
    """
    Asserts that a run pairs its settings with the commit its clone has
    checked out, the digest of the clone's `uv.lock`, and the running
    platform.
    """
    assert Run.from_checkout(clone, Settings()) == Run(
        commit      = clone.commit,
        lock_digest = clone.lock_digest,
        platform    = platform(),
        settings    = Settings()
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
