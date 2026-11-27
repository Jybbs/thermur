"""
Pins which `AUDIT_IGNORE` ids `lock:scan` passes to `bun audit --ignore`,
and how it fails before the audit on an id a release fixes, naming the
entry and the patched version, or on an id `gh` cannot read, each against a
stand-in `gh`.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import run_lock
from pathlib          import Path
from pytest           import MonkeyPatch, mark, param
from syrupy.assertion import SnapshotAssertion

MICROMATCH = "GHSA-952p-6rrq-rcjv"
PICOMATCH  = "GHSA-fc5q-39xc-vjpf"


@mark.parametrize(
    "ignored",
    [
        param([], id="empty-list"),
        param([PICOMATCH], id="one-unfixed"),
        param([PICOMATCH, MICROMATCH], id="two-unfixed")
    ]
)
def test_the_scan_sets_aside_each_advisory_no_release_fixes(
    lock_copy   : Path,
    ignored     : list[str],
    monkeypatch : MonkeyPatch,
    snapshot    : SnapshotAssertion,
    stand_in    : Callable[[str, str], StandIn]
):
    """
    Asserts that `lock:scan` reads each `AUDIT_IGNORE` id through `gh api`
    and, while `gh` lists no `first_patched_version` for it, passes it to
    `bun audit` as an `--ignore`, running the plain audit over an empty
    list.
    """
    monkeypatch.setenv("AUDIT_IGNORE", " ".join(ignored))
    programs = {program: stand_in(program, "exit 0") for program in ("bun", "gh")}

    assert run_lock(lock_copy, "scan").returncode == 0
    assert "\n".join(programs["gh"].commands) == snapshot
    assert programs["bun"].commands == [
        "bun audit --cwd docs" + "".join(
            f" --ignore={advisory}" for advisory in ignored
        )
    ]


@mark.parametrize(
    ("gh", "reported"),
    [
        param(
            f'[ "$4" = /advisories/{MICROMATCH} ] && echo "micromatch 4.0.8"\nexit 0',
            f"{MICROMATCH} is fixed in micromatch 4.0.8, so delete its"
            " AUDIT_IGNORE entry in .mise/config.toml and lift to the patched release",
            id = "fixed"
        ),
        param(
            f'[ "$4" = /advisories/{MICROMATCH} ] && printf "a 1.0\\nb 2.0\\n"\nexit 0',
            f"{MICROMATCH} is fixed in a 1.0, b 2.0, so delete its"
            " AUDIT_IGNORE entry in .mise/config.toml and lift to the patched release",
            id = "fixed-in-two-packages"
        ),
        param(
            'echo "HTTP 404: Not Found" >&2\nexit 1',
            "HTTP 404: Not Found",
            id = "unreadable"
        )
    ]
)
def test_the_scan_fails_before_auditing_on_an_entry_it_cannot_set_aside(
    lock_copy   : Path,
    gh          : str,
    monkeypatch : MonkeyPatch,
    reported    : str,
    stand_in    : Callable[[str, str], StandIn]
):
    """
    Asserts that an `AUDIT_IGNORE` id for which `gh` lists a patched release
    fails `lock:scan`, which names the entry to delete and the release to
    lift to, and that a `gh api` read failing fails it too, each before the
    task audits `docs/bun.lock`.
    """
    monkeypatch.setenv("AUDIT_IGNORE", f"{PICOMATCH} {MICROMATCH}")
    stand_in("gh", gh)
    bun = stand_in("bun", "exit 0")

    result = run_lock(lock_copy, "scan")

    assert (result.returncode, result.stderr.strip(), bun.commands) == (
        1, reported,
        []
    )
