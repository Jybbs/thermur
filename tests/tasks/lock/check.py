"""
Pins the path `lock:check` takes for each answer `uv`, `mise`, and `bun`
give it, each case running a copy of the task beside a lockfile of its own,
and what the real mise answers for the committed lockfile online, for one
missing a platform entry, a checksum, or a pin's entry offline, and for a
pin moved with its entry to a release that does not exist, where
`lock:check` fails on each pinned version `mise ls-remote` does not list.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import run_lock
from pathlib          import Path
from pytest           import MonkeyPatch, mark, param
from re               import escape, sub
from shlex            import quote
from shutil           import which
from tomllib          import loads

CONFIG      = ".mise/config.toml"
FROZEN      = "error: lockfile had changes, but lockfile is frozen"
LOCK        = ".mise/mise.lock"
LYCHEE      = '[tools."aqua:lycheeverse/lychee"."platforms.macos-x64"]'
MISSING = "No lockfile URL found for uv@0.12.22 on platform macos-arm64 (--locked mode)"
PIN         = "uv  0.13.8 (missing)  .mise/config.toml  0.13.8"
STALE       = "zizmor@0.0.1 is not in the lockfile"
UNREACHABLE = "mise ERROR unable to fetch versions for uv: error sending request"
UNREADABLE  = "mise ERROR failed to parse .mise/config.toml"


def pinning(copy: Path) -> dict[str, str]:
    """
    Reads `.mise/config.toml` and `.mise/mise.lock` in the copy, keyed by
    each file's path under the copy's root.
    """
    return {name: (copy / name).read_text() for name in (CONFIG, LOCK)}


def test_a_current_lock_passes(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch,
    stand_in    : Callable[[str, str], StandIn]
):
    """
    Asserts that the task exits `0` where `uv.lock` is current, `mise lock
    --check` passes, the dry run installs every tool the project pins under
    the project's lock scope alone, whatever scope the environment names,
    and `mise ls-remote` lists the version each pin names, while the task
    prints nothing and leaves `.mise/mise.lock` the file it found.
    """
    monkeypatch.setenv("MISE_LOCKED_SCOPES", "global")
    lock = lock_copy / LOCK
    held = (lock.read_bytes(), lock.stat().st_ino)
    mise = stand_in(
        "mise",
        '[ "$1" = install ] && echo "$MISE_LOCKED_SCOPES" > scopes\n'
        f'[ "$1" = ls ] && echo "{PIN}"\n'
        '[ "$1" = ls-remote ] && printf "0.13.7\\n0.13.8\\n"\n'
        "exit 0"
    )

    result = run_lock(lock_copy)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert (lock.read_bytes(), lock.stat().st_ino) == held
    assert ((lock_copy / "scopes").read_text(), mise.commands) == (
        "project\n",
        [
            "mise lock --check", "mise install --dry-run --force --locked",
            "mise ls --no-header --project", "mise ls-remote uv"
        ]
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


@mark.usefixtures("mise")
@mark.parametrize(
    ("program", "script", "reported"),
    [
        param(
            "mise",
            f'[ "$1" = lock ] && echo "{STALE}" >&2 && exit 1\nexit 0',
            STALE,
            id = "failed-check"
        ),
        param(
            "mise",
            f'[ "$1" = ls ] && echo "{UNREADABLE}" >&2 && exit 1\nexit 0',
            UNREADABLE,
            id = "failed-listing"
        ),
        param(
            "mise",
            f'[ "$1" = install ] && echo "{MISSING}" >&2 && exit 1\nexit 0',
            MISSING,
            id = "missing-entry"
        ),
        param(
            "mise",
            f'[ "$1" = ls ] && echo "{PIN}"\n'
            f'[ "$1" = ls-remote ] && echo "{UNREACHABLE}" >&2 && exit 1\n'
            "exit 0",
            UNREACHABLE,
            id = "unreachable-host"
        ),
        param(
            "mise",
            f'[ "$1" = ls ] && echo "{PIN}"\nexit 0',
            "mise ls-remote lists no version of uv, as when mise runs offline",
            id = "no-version-list"
        ),
        param("bun", f'echo "{FROZEN}" >&2\nexit 1', FROZEN, id="site-lockfile")
    ]
)
def test_the_check_fails_printing_what_each_program_reports(
    lock_copy : Path,
    program   : str,
    reported  : str,
    script    : str,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that a `mise lock --check` exiting nonzero, a dry run finding
    no lockfile entry for a pinned tool on the running platform, a `mise
    ls` failing to list the pins, a `mise ls-remote` reaching no host, and
    a frozen dry run of `bun install` finding `docs/bun.lock` lagging its
    manifest each fail the task, which prints what the program reported, and
    that a `mise ls-remote` listing no version at all, as it does offline
    before it caches a list, fails the task naming the tool.
    """
    stand_in(program, script)

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr.strip()) == (1, reported)


def test_each_pin_mise_ls_remote_does_not_list_fails_naming_it(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that the task fails on every pin whose version `mise ls-remote`
    lists no line of its own, naming each tool and its version, rather than
    stopping at the first or passing a version that only opens a listed one.
    """
    mise = stand_in(
        "mise",
        '[ "$1" = ls ] && printf "%s\\n" "aqua:lycheeverse/lychee  0.99.9"'
        ' "bun  1.4.4" "zizmor  1.32"\n'
        '[ "$1" = ls-remote ] && printf "0.99.90\\n1.4.4\\n1.32.0\\n"\n'
        "exit 0"
    )

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr.splitlines()) == (
        1,
        [
            "aqua:lycheeverse/lychee pins 0.99.9, which mise ls-remote does not list",
            "zizmor pins 1.32, which mise ls-remote does not list"
        ]
    )
    assert [command for command in mise.commands if "ls-remote" in command] == [
        "mise ls-remote aqua:lycheeverse/lychee", "mise ls-remote bun",
        "mise ls-remote zizmor"
    ]


@mark.parametrize(
    ("path", "pattern", "replacement", "reported"),
    [
        param(
            LOCK,
            rf"{escape(LYCHEE)}\n(?:.+\n)+\n?",
            "",
            "No lockfile URL found for aqua:lycheeverse/lychee@{lychee} on platform"
            " macos-x64 (--locked mode)",
            id = "lychee-entry"
        ),
        param(
            LOCK,
            r'\[tools\.zizmor\."platforms\.windows-x64-baseline"\]\n(?:.+\n)+\n?',
            "",
            "No lockfile URL found for zizmor@{zizmor} on platform"
            " windows-x64-baseline (--locked mode)",
            id = "zizmor-entry"
        ),
        param(
            LOCK,
            rf"({escape(LYCHEE)}\n)checksum = .+\n",
            r"\1",
            "No lockfile checksum found for aqua:lycheeverse/lychee@{lychee} on"
            " platform macos-x64",
            id = "checksum"
        ),
        param(
            CONFIG,
            r'(?m)^(zizmor\s*=\s*)".+"$',
            r'\1"0.0.1"',
            "zizmor@0.0.1 is not in the lockfile",
            id = "pin"
        )
    ]
)
def test_the_real_mise_fails_offline_on_what_a_locked_install_rejects(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch,
    path        : str,
    pattern     : str,
    replacement : str,
    reported    : str
):
    """
    Asserts that the real `mise lock --check`, reaching no host through
    a proxy that refuses every connection, fails the task on a lockfile
    missing one platform entry or one entry's checksum, and on a pin the
    lockfile holds no entry for, naming the tool, its version, and the
    platform, and that the task leaves both files as it found them.

    `MISE_HTTP_RETRIES` set to `0` makes each refused request fail at once
    rather than after mise's retries.
    """
    found       = pinning(lock_copy)
    pinned      = loads(found[CONFIG])["tools"]
    found[path] = sub(pattern, replacement, found[path], count=1)
    written(lock_copy, found)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("MISE_HTTP_RETRIES", "0")
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(lock_copy))

    result = run_lock(lock_copy)

    assert (result.returncode, pinning(lock_copy)) == (1, found)
    assert reported.format(
        lychee = pinned["aqua:lycheeverse/lychee"],
        zizmor = pinned["zizmor"]
    ) in result.stderr


@mark.parametrize(
    "online",
    [param(False, id="stood-in"), param(True, id="online", marks=mark.network)]
)
def test_a_pin_moved_with_its_lock_entry_to_no_release_fails(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch,
    online      : bool,
    stand_in    : Callable[[str, str], StandIn]
):
    """
    Asserts that the lychee pin moved, with every version and URL its lock
    entry holds, to `0.99.9`, a release lychee never published, which the
    real `mise lock --check` and the `--locked` dry run both pass, fails
    the task, which names the tool and the version and leaves both files
    as it moved them. The stood-in case answers `mise ls-remote` with every
    version `.mise/config.toml` pins before the move and runs the real
    `mise` for the rest, while the online case reads each real version list.
    """
    found  = pinning(lock_copy)
    pinned = loads(found[CONFIG])["tools"]
    held   = written(
        copy  = lock_copy,
        texts = {
            name: text.replace(pinned["aqua:lycheeverse/lychee"], "0.99.9")
            for name, text in found.items()
        }
    )
    if not online:
        listed = quote("\n".join(pinned.values()))
        real   = quote(which("mise"))
        stand_in(
            "mise",
            f'[ "$1" = ls-remote ] && echo {listed} && exit 0\nexec {real} "$@"'
        )
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(lock_copy))

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr) == (
        1, "aqua:lycheeverse/lychee pins 0.99.9, which mise ls-remote does not list\n"
    )
    assert pinning(lock_copy) == held


@mark.network
def test_the_committed_lock_passes_the_check_online(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the real `mise`, checking every platform `.mise/mise.lock`
    holds, running the dry run, and reading each pinned tool's version list
    over the network, passes the task on the committed lockfile, printing
    nothing.
    """
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(lock_copy))

    result = run_lock(lock_copy)

    assert (result.returncode, result.stderr) == (0, "")


def written(copy: Path, texts: dict[str, str]) -> dict[str, str]:
    """
    Writes each text over the file its path names under the copy's root,
    returning the texts.
    """
    for name, text in texts.items():
        (copy / name).write_text(text)
    return texts
