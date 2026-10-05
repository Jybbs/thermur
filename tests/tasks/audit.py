"""
Pins the verdict `repo:audit` exits with and what it prints, over a copy of
the checkout as it stands, over one that breaks every check, and over one
whose manifest fails to validate.
"""

from collections.abc  import Callable
from common.edits     import edit
from pytest           import CaptureFixture, MonkeyPatch, fixture, raises
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

from thermur.repo.checkout import Checkout


@fixture
def audited(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
) -> Callable[[], tuple[int, list[str]]]:
    """
    Enters the copy of the checkout, so a case edits it and then runs the
    audit over it.

    Returns:
        A runner returning the status the audit exits with and the lines it
        prints.
    """
    monkeypatch.chdir(checkout.root)

    def run() -> tuple[int, list[str]]:
        """
        Runs the audit's `main` over the copy.
        """
        with raises(SystemExit) as exited:
            load_task(".mise/tasks/repo/audit.py").main()

        return exited.value.code, capsys.readouterr().out.splitlines()

    return run


def test_a_broken_checkout_fails_naming_each_finding(
    audited  : Callable[[], tuple[int, list[str]]],
    checkout : Checkout,
    snapshot : SnapshotAssertion
):
    """
    Asserts that the audit exits `1` over a checkout breaking every check,
    printing one annotation per finding in the order the checks run.
    """
    root = checkout.root
    (root / ".mise/bin/pytest").unlink()
    (root / ".mise/bin/pytest").symlink_to("prose")
    (root / ".mise/tasks/py/test").write_text("#!/bin/sh\nuv run pytest\n")
    (root / "LICENSE").write_text("BSD License\n")
    (root / "src/thermur").rename(root / "thermur")
    (root / "src/thermur").mkdir()
    edit(checkout, "pyproject.toml", new='"hatchling>=1.32"', old='"hatchling==1.32.4"')

    status, printed = audited()

    assert status == 1
    assert "\n".join(printed) == snapshot


def test_a_clean_checkout_passes_printing_nothing(
    audited: Callable[[], tuple[int, list[str]]]
):
    """
    Asserts that the audit exits `0` and prints nothing over a copy of the
    checkout as it stands.
    """
    assert audited() == (0, [])


def test_a_document_failing_validation_prints_each_error_once(
    audited  : Callable[[], tuple[int, list[str]]],
    checkout : Checkout
):
    """
    Asserts that a manifest missing a field, which every check reading the
    manifest reports, fails the audit with one annotation for that field.
    """
    edit(checkout, "pyproject.toml", new="", old='requires-python = ">=3.14"\n')

    assert audited() == (
        1,
        [
            "::error file=pyproject.toml::`project.requires-python` in "
            "`pyproject.toml`: Field required"
        ]
    )
