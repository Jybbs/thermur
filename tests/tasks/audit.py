"""
Pins the verdict `repo:audit` exits with and what it prints, over a copy of
the checkout as it stands, over one that breaks every check, over one whose
manifest fails to validate, and over one whose only findings are warnings.
"""

from collections.abc  import Callable
from common.edits     import append, edit, task
from pytest           import CaptureFixture, MonkeyPatch, fixture, raises
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Issue, Level


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
    pins     : dict[str, str],
    snapshot : SnapshotAssertion
):
    """
    Asserts that the audit exits `1` over a checkout breaking every check,
    printing one annotation per finding in the order the checks run. Where a
    finding names two pins, the case sets both, so the printed text names no
    release a lift moves.
    """
    root   = checkout.root
    action = ".github/actions/provision/action.yml"
    draft  = ".github/workflows/draft.yml"
    warm   = ".github/workflows/warm.yml"
    (root / ".mise/bin/pytest").unlink()
    (root / ".mise/bin/pytest").symlink_to("prose")
    (root / ".mise/tasks/py/test").write_text("#!/bin/sh\nuv run pytest\nbunx knip\n")
    (root / "LICENSE").write_text("BSD License\n")
    (root / "src/thermur").rename(root / "thermur")
    (root / "src/thermur").mkdir()
    edit(
        checkout,
        "pyproject.toml",
        new = '"hatchling>=1"',
        old = f'"hatchling=={pins["hatchling"]}"'
    )
    edit(checkout, ".github/release.yml", new='["🐛 bug"]', old='["🐞 bug"]')
    edit(
        checkout,
        ".github/CONTRIBUTING.md",
        new = "`mise run lock:scan`",
        old = "`mise scan`"
    )
    edit(checkout, action, new="default     : &depth '1'", old="default     : '1'")
    edit(checkout, action, new="default     : *depth", old="default     : 'false'")
    for workflow in ("ci", "site"):
        edit(
            checkout,
            f".github/workflows/{workflow}.yml",
            new = "",
            old = "    name    : 📡 Readout\n"
        )
    edit(
        checkout,
        ".mise/tasks/gha/brief.py",
        new = "==0.0.1",
        old = f"=={pins['minijinja']}"
    )
    edit(
        checkout,
        "uv.lock",
        new = 'name = "minijinja"\nversion = "9.9.9"',
        old = f'name = "minijinja"\nversion = "{pins["minijinja"]}"'
    )
    edit(
        checkout,
        "docs/package.json",
        new = '"@vitest/coverage-v8" : "0.0.1"',
        old = f'"@vitest/coverage-v8" : "{pins["vitest"]}"'
    )
    edit(
        checkout,
        "docs/package.json",
        new = '"vitest"              : "0.0.2"',
        old = f'"vitest"              : "{pins["vitest"]}"'
    )
    edit(
        checkout,
        action,
        new = "actions/checkout@1f1f",
        old = f"actions/checkout@{pins['checkout']}"
    )
    edit(
        checkout,
        "docs/.vitepress/lib/rendered/identity.json",
        new = '"alpha": 0.5',
        old = '"alpha": 0.64'
    )
    edit(
        checkout,
        draft,
        new = "run  : mise run gha:brief",
        old = "run  : ./.mise/tasks/gha/cut.py"
    )
    edit(
        checkout = checkout,
        new      = "uses : actions/checkout@0f0f\n\n      - name : Cut",
        old = "uses : $/.github/actions/provision\n        with : { depth: '2' }\n\n"
        "      - name : Cut",
        path = draft
    )
    edit(checkout, warm, new="&image ubuntu-24.04", old="&image ubuntu-26.04")
    edit(
        checkout,
        warm,
        new = "",
        old = "          - name  : 🎒 Kit · Workflows\n            tools : zizmor\n"
    )
    edit(checkout, warm, new="]", old=", .mise/mise.lock]")
    append(
        checkout,
        "docs/pages/index.md",
        "\nA study[^smith2020] beside[^cavagna2010].\n"
    )
    task(checkout, "py/sweep", '#!/bin/sh\n#MISE alias = "audit"\ntrue\n')

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
    checkout : Checkout,
    pins     : dict[str, str]
):
    """
    Asserts that a manifest missing a field, which every check reading the
    manifest reports, fails the audit with one annotation for that field.
    """
    edit(
        checkout,
        "pyproject.toml",
        new = "",
        old = f'requires-python = ">={pins["floor"]}"\n'
    )

    assert audited() == (
        1,
        [
            "::error file=pyproject.toml::`project.requires-python` in "
            "`pyproject.toml`: Field required"
        ]
    )


def test_warnings_alone_pass_printing_each(
    audited     : Callable[[], tuple[int, list[str]]],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the audit exits `0` where every finding is a warning,
    printing each as a warning annotation.
    """
    warned = Issue(message="m", severity=Level.WARNING, task="py:test")
    monkeypatch.setattr(Checkout, "issues", property(lambda checkout: [warned]))

    assert audited() == (
        0,
        [
            "::warning file=.mise/tasks/py/test::`mise tasks validate` reports `m` "
            "on `py:test`"
        ]
    )
