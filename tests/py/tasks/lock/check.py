"""
Pins the path `lock:check` takes for each answer `uv`, `mise`, and `bun`
give it, each case running a copy of the task beside a lockfile of its own,
and what the real mise answers for the committed lockfile online and for one
missing a platform entry offline, where `lock:check` fails on a `mise lock`
exiting nonzero or counting a skipped platform and on a lockfile in which a
tool holds no entry for a platform another tool carries.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import SUMMARY, run_lock
from pathlib          import Path
from pytest           import MonkeyPatch, mark, param
from re               import escape, sub

CHECKED = "mise lock    ✓ 72 targets checked"
FAILED  = "mise ERROR error sending request"
LYCHEE  = '[tools."aqua:lycheeverse/lychee"."platforms.linux-x64"]'
MISSING = "No lockfile URL found for uv@0.12.22 on platform macos-arm64 (--locked mode)"


@mark.parametrize(
    "summary",
    [
        param(SUMMARY.format(skipped=0), id="no-skipped-platform"),
        param(CHECKED, id="the-targets-checked")
    ]
)
def test_a_current_lock_passes(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn],
    summary   : str
):
    """
    Asserts that the task exits `0` where `uv.lock` is current, `mise lock`
    writes every platform and changes nothing, reporting no skipped platform
    or only the targets it checked, and the dry run installs every tool the
    project pins under the project's lock scope alone, while the task prints
    nothing and leaves `.mise/mise.lock` the file it found.
    """
    lock = lock_copy / ".mise/mise.lock"
    held = (lock.read_bytes(), lock.stat().st_ino)
    stand_in(
        "mise",
        f'[ "$1" = lock ] && echo "{summary}"\n'
        'echo "$MISE_LOCKED_SCOPES $*" >> install\n'
        'echo "mise uv@0.12.22 would install"'
    )

    result = run_lock(lock_copy)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert (lock.read_bytes(), lock.stat().st_ino) == held
    assert (lock_copy / "install").read_text().splitlines()[-1] == (
        "project install --dry-run --force --locked"
    )


def test_a_failing_uv_check_stops_before_mise_runs(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that a `uv.lock` lagging its manifest fails the task with `uv`'s
    status before `mise` runs at all.
    """
    stand_in("uv", "exit 1")
    stand_in("mise", 'echo "$*" >> ran')

    assert (run_lock(lock_copy).returncode, (lock_copy / "ran").exists()) == (1, False)


def test_a_rewritten_lock_fails_and_is_restored(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that a `mise lock` rewriting `.mise/mise.lock` fails the task,
    which prints the difference and leaves the file as it found it, byte for
    byte and mode for mode.
    """
    lock = lock_copy / ".mise/mise.lock"
    held = (lock.read_bytes(), lock.stat().st_mode)
    stand_in(
        "mise",
        'echo "drifted = true" >> .mise/mise.lock\n'
        "chmod 600 .mise/mise.lock\n"
        f'echo "{SUMMARY.format(skipped=0)}"'
    )

    result = run_lock(lock_copy)

    assert (result.returncode, "+drifted = true" in result.stderr) == (1, True)
    assert (lock.read_bytes(), lock.stat().st_mode) == held


def test_a_site_lockfile_lagging_its_manifest_fails_the_check(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that a `docs/bun.lock` lagging `docs/package.json`, which a
    frozen dry run of `bun install` reports by exiting `1`, fails the task
    once `uv.lock` and `.mise/mise.lock` both pass.
    """
    stand_in(
        "bun",
        'echo "error: lockfile had changes, but lockfile is frozen" >&2\nexit 1'
    )
    stand_in("mise", f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}"\nexit 0')

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr.strip()) == (
        1, "error: lockfile had changes, but lockfile is frozen"
    )


@mark.parametrize(
    ("mise", "reported"),
    [
        param(
            f'echo "{SUMMARY.format(skipped=3)}"',
            SUMMARY.format(skipped=3),
            id = "skipped-platform"
        ),
        param(
            f'echo "{SUMMARY.format(skipped=10)}"',
            SUMMARY.format(skipped=10),
            id = "ten-skipped-platforms"
        ),
        param(
            f'[ "$1" = lock ] && echo "{FAILED}" >&2 && exit 1\nexit 0',
            FAILED,
            id = "failed-lock"
        ),
        param(
            f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}" && exit 0\n'
            f'echo "{MISSING}" >&2\n'
            "exit 1",
            MISSING,
            id = "missing-entry"
        )
    ],
    indirect = ["mise"]
)
def test_the_check_fails_printing_what_mise_reports(
    lock_copy : Path,
    mise      : StandIn,
    reported  : str
):
    """
    Asserts that a `mise lock` counting a skipped platform, which exits `0`,
    a `mise lock` exiting nonzero, and a dry run finding no lockfile entry
    for a pinned tool on the running platform each fail the task, which
    prints what mise reported.
    """
    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr.strip()) == (1, reported)


def test_a_tool_missing_a_platform_another_carries_fails_naming_both(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that a lockfile in which one tool holds no entry for a platform
    another tool's entries carry fails the task, which names the tool
    and the platform, even where `mise lock` exits `0`, counts no skipped
    platform, and leaves the file as it found it.
    """
    lock = lock_copy / ".mise/mise.lock"
    lock.write_text(sub(rf"{escape(LYCHEE)}\n(?:.+\n)+\n?", "", lock.read_text()))
    stand_in("mise", f'echo "{CHECKED}"')

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr.strip()) == (
        1, "aqua:lycheeverse/lychee has no linux-x64 entry in .mise/mise.lock"
    )


@mark.parametrize(
    "table",
    [
        param(LYCHEE, id="lychee"),
        param('[tools.zizmor."platforms.windows-x64-baseline"]', id="zizmor")
    ]
)
def test_an_offline_lock_fails_the_check(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch,
    table       : str
):
    """
    Asserts that the real `mise lock`, reaching no release host through
    a proxy that refuses every connection, fails the task on a lockfile
    missing one platform entry, which it cannot fetch, and that the task
    leaves the lockfile as it found it.

    `MISE_HTTP_RETRIES` set to `0` makes each refused request fail at once
    rather than after mise's retries.
    """
    lock    = lock_copy / ".mise/mise.lock"
    trimmed = sub(rf"{escape(table)}\n(?:.+\n)+\n?", "", lock.read_text())
    lock.write_text(trimmed)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("MISE_HTTP_RETRIES", "0")
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(lock_copy))

    assert (run_lock(lock_copy).returncode, lock.read_text()) == (1, trimmed)


@mark.network
def test_the_committed_lock_passes_the_check_online(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the real `mise`, locking every platform `.mise/mise.lock`
    holds and running the dry run, passes the task on the committed
    lockfile, printing nothing. The release `min_version` names fetches each
    platform entry over the network to lock it.
    """
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(lock_copy))

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr) == (0, "")
