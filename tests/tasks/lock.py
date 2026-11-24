"""
Pins the path `lock:check` takes for each answer `uv`, `mise`, and `bun`
give it, each case running a copy of the task beside a lockfile of its
own, the lockfile beside each task script declaring its own dependencies
that every `lock` task checks, audits, or writes, and what the real mise
answers for the committed lockfile online, for one missing a platform entry,
a checksum, or a pin's entry offline, and for a pin moved with its entry
to a release that does not exist, where `lock:check` fails on each pinned
version `mise ls-remote` does not list. It also pins which `AUDIT_IGNORE`
ids `lock:scan` passes to `bun audit --ignore`, and how it fails before the
audit on an id a release fixes, naming the entry and the patched version, or
on an id `gh` cannot read, each against a stand-in `gh`. Last, it pins that
`lock:sync`, running the real `uv` against a stand-in registry, leaves out
of `uv.lock` a release newer than the window `[tool.uv]` declares, and fails
before locking on a window written other than as whole days.
"""

from collections.abc  import Callable
from common.registry  import serving
from common.stand_ins import StandIn
from datetime         import UTC, datetime
from os               import environ, pathsep
from pathlib          import Path
from pytest           import Config, MonkeyPatch, fixture, mark, param
from re               import escape, sub
from shlex            import quote
from shutil           import which
from subprocess       import CompletedProcess, run
from syrupy.assertion import SnapshotAssertion
from tomllib          import loads

from thermur.repo.checkout import Checkout

CONFIG     = ".mise/config.toml"
LOCK       = ".mise/mise.lock"
LYCHEE     = '[tools."aqua:lycheeverse/lychee"."platforms.macos-x64"]'
MICROMATCH = "GHSA-952p-6rrq-rcjv"
MISSING = "No lockfile URL found for uv@0.12.22 on platform macos-arm64 (--locked mode)"
PICOMATCH   = "GHSA-fc5q-39xc-vjpf"
PIN         = "uv  0.13.8 (missing)  .mise/config.toml  0.13.8"
STALE       = "zizmor@0.0.1 is not in the lockfile"
UNREACHABLE = "mise ERROR unable to fetch versions for uv: error sending request"


def executed(copy: Path, task: str = "check") -> CompletedProcess:
    """
    Runs the copy of the `lock` task named, `lock:check` unless given,
    through its own shebang from the root of the copy, capturing what it
    prints.
    """
    return run(
        [copy / ".mise/tasks/lock" / task],
        capture_output = True,
        cwd            = copy,
        text           = True
    )


def pinning(copy: Path) -> dict[str, str]:
    """
    Reads `.mise/config.toml` and `.mise/mise.lock` in the copy, keyed by
    each file's path under the copy's root.
    """
    return {name: (copy / name).read_text() for name in (CONFIG, LOCK)}


@fixture
def copy(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn],
    tmp_path     : Path
) -> Path:
    """
    Copies the `lock` tasks, the library they source, `.mise/config.toml`,
    `.mise/mise.lock`, and `pyproject.toml` into `tmp_path`, keeping
    each file's mode, names the copy as the root a task sources its
    library from, clears the `AUDIT_IGNORE` list `mise test` sets from
    `.mise/config.toml`, and puts a stand-in `uv` that reports `uv.lock`
    current, a stand-in `bun` that reports `docs/bun.lock` current, and a
    stand-in `gh` that lists no patched release ahead on the path.

    Returns:
        The copy's root.
    """
    (tmp_path / ".mise/tasks").mkdir(parents=True)
    for path in (
        CONFIG, ".mise/lib", LOCK, ".mise/tasks/lock", "pyproject.toml"
    ):
        (pytestconfig.rootpath / path).copy(tmp_path / path, preserve_metadata=True)

    monkeypatch.delenv("AUDIT_IGNORE", raising=False)
    monkeypatch.setenv("MISE_PROJECT_ROOT", str(tmp_path))
    stand_in("bun", "exit 0")
    stand_in("gh", "exit 0")
    stand_in("uv", "exit 0")
    return tmp_path


def test_a_current_lock_passes(
    copy        : Path,
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
    lock = copy / LOCK
    held = (lock.read_bytes(), lock.stat().st_ino)
    mise = stand_in(
        "mise",
        '[ "$1" = install ] && echo "$MISE_LOCKED_SCOPES" > scopes\n'
        f'[ "$1" = ls ] && echo "{PIN}"\n'
        '[ "$1" = ls-remote ] && printf "0.13.7\\n0.13.8\\n"\n'
        "exit 0"
    )

    result = executed(copy)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert (lock.read_bytes(), lock.stat().st_ino) == held
    assert ((copy / "scopes").read_text(), mise.commands) == (
        "project\n",
        [
            "mise lock --check", "mise install --dry-run --force --locked",
            "mise ls --no-header --project", "mise ls-remote uv"
        ]
    )


def test_a_failing_uv_check_stops_before_mise_runs(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn]
):
    """
    Asserts that a `uv.lock` lagging its manifest fails the task with `uv`'s
    status before `mise` runs at all.
    """
    stand_in("uv", "exit 1")
    stand_in("mise", 'echo "$*" >> ran')

    assert (executed(copy).returncode, (copy / "ran").exists()) == (1, False)


def test_a_site_lockfile_lagging_its_manifest_fails_the_check(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn]
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
    stand_in("mise", "exit 0")

    result = executed(copy)

    assert (result.returncode, result.stderr.strip()) == (
        1, "error: lockfile had changes, but lockfile is frozen"
    )


@mark.parametrize(
    ("mise", "reported"),
    [
        param(
            f'[ "$1" = lock ] && echo "{STALE}" >&2 && exit 1\nexit 0',
            STALE,
            id = "failed-check"
        ),
        param(
            f'[ "$1" = install ] && echo "{MISSING}" >&2 && exit 1\nexit 0',
            MISSING,
            id = "missing-entry"
        ),
        param(
            f'[ "$1" = ls ] && echo "{PIN}"\n'
            f'[ "$1" = ls-remote ] && echo "{UNREACHABLE}" >&2 && exit 1\n'
            "exit 0",
            UNREACHABLE,
            id = "unreachable-host"
        ),
        param(
            f'[ "$1" = ls ] && echo "{PIN}"\nexit 0',
            "mise ls-remote lists no version of uv, as when mise runs offline",
            id = "no-version-list"
        )
    ],
    indirect = ["mise"]
)
def test_the_check_fails_printing_what_mise_reports(
    copy     : Path,
    mise     : StandIn,
    reported : str
):
    """
    Asserts that a `mise lock --check` exiting nonzero, a dry run finding
    no lockfile entry for a pinned tool on the running platform, and a `mise
    ls-remote` reaching no host each fail the task, which prints what mise
    reported, and that a `mise ls-remote` listing no version at all, as it
    does offline before it caches a list, fails the task naming the tool.
    """
    result = executed(copy)

    assert (result.returncode, result.stderr.strip()) == (1, reported)


def test_each_pin_mise_ls_remote_does_not_list_fails_naming_it(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn]
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

    result = executed(copy)

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
    copy        : Path,
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
    held       = pinning(copy)
    pinned     = loads(held[CONFIG])["tools"]
    held[path] = sub(pattern, replacement, held[path], count=1)
    for name, text in held.items():
        (copy / name).write_text(text)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("MISE_HTTP_RETRIES", "0")
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(copy))

    result = executed(copy)

    assert (result.returncode, pinning(copy)) == (1, held)
    assert reported.format(
        lychee = pinned["aqua:lycheeverse/lychee"],
        zizmor = pinned["zizmor"]
    ) in result.stderr


@mark.parametrize(
    "online",
    [param(False, id="stood-in"), param(True, id="online", marks=mark.network)]
)
def test_a_pin_moved_with_its_lock_entry_to_no_release_fails(
    copy        : Path,
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
    pinned = loads((copy / CONFIG).read_text())["tools"]
    held   = {
        name: text.replace(pinned["aqua:lycheeverse/lychee"], "0.99.9")
        for name, text in pinning(copy).items()
    }
    for name, text in held.items():
        (copy / name).write_text(text)
    if not online:
        listed = quote("\n".join(pinned.values()))
        real   = quote(which("mise"))
        stand_in(
            "mise",
            f'[ "$1" = ls-remote ] && echo {listed} && exit 0\nexec {real} "$@"'
        )
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(copy))

    result = executed(copy)

    assert (result.returncode, result.stderr) == (
        1, "aqua:lycheeverse/lychee pins 0.99.9, which mise ls-remote does not list\n"
    )
    assert pinning(copy) == held


@mark.network
def test_the_committed_lock_passes_the_check_online(
    copy        : Path,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the real `mise`, checking every platform `.mise/mise.lock`
    holds, running the dry run, and reading each pinned tool's version list
    over the network, passes the task on the committed lockfile, printing
    nothing.
    """
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(copy))

    result = executed(copy)

    assert (result.returncode, result.stderr) == (0, "")


@mark.parametrize(
    ("task", "ran", "bun_command"),
    [
        param(
            "check",
            ["lock --check", "lock --check --script {script}"],
            "install --cwd docs --dry-run --frozen-lockfile",
            id = "check"
        ),
        param(
            "scan",
            [
                "audit --preview-features audit-command",
                "audit --preview-features audit-command --script {script}"
            ],
            "audit --cwd docs",
            id = "scan"
        ),
        param(
            "sync",
            ["lock --no-locked", "lock --no-locked --script {script}"],
            "install --cwd docs --lockfile-only --minimum-release-age={age}",
            id = "sync"
        )
    ]
)
def test_each_lock_task_reaches_every_lockfile(
    bun_command : str,
    copy        : Path,
    ran         : list[str],
    stand_in    : Callable[[str, str], StandIn],
    task        : str
):
    """
    Asserts that each `lock` task runs its `uv` command over `uv.lock` and
    then over the lockfile beside each task script whose inline metadata
    opens a `script` block, leaving out a script declaring none, and runs
    its `bun` command over `docs/bun.lock` once, `lock:sync` passing the
    window `[tool.uv]` declares in whole seconds as Bun's minimum release
    age.
    """
    age    = int(Checkout(root=copy).manifest.window.total_seconds())
    script = ".mise/tasks/gha/declared.py"
    (copy / ".mise/tasks/gha").mkdir()
    (copy / script).write_text('# /// script\n# dependencies = ["minijinja"]\n# ///\n')
    (copy / ".mise/tasks/gha/plain.py").write_text('"""\nDeclares nothing.\n"""\n')
    programs = {program: stand_in(program, "exit 0") for program in ("bun", "uv")}
    stand_in("mise", "exit 0")

    assert executed(copy, task).returncode == 0
    assert programs["uv"].commands == [
        f"uv {command.format(script=script)}" for command in ran
    ]
    assert programs["bun"].commands == [f"bun {bun_command.format(age=age)}"]


@mark.parametrize(
    "ignored",
    [
        param([], id="empty-list"),
        param([PICOMATCH], id="one-unfixed"),
        param([PICOMATCH, MICROMATCH], id="two-unfixed")
    ]
)
def test_the_scan_sets_aside_each_advisory_no_release_fixes(
    copy        : Path,
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

    assert executed(copy, "scan").returncode == 0
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
    copy        : Path,
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

    result = executed(copy, "scan")

    assert (result.returncode, result.stderr.strip(), bun.commands) == (
        1, reported,
        []
    )


@mark.enable_socket
@mark.allow_hosts(["127.0.0.1"])
def test_the_sync_leaves_a_release_newer_than_the_window_out_of_uv_lock(
    copy        : Path,
    monkeypatch : MonkeyPatch,
    stand_in    : Callable[[str, str], StandIn]
):
    """
    Asserts that `lock:sync`, resolving the package `aged` through the real
    `uv` against a registry publishing `1.0.0` twice the window `[tool.uv]`
    declares ago and `2.0.0` half of it ago, locks `1.0.0` in `uv.lock`,
    reading the window the worktree's `pyproject.toml` declares.
    """
    window   = Checkout(root=copy).manifest.window
    now      = datetime.now(UTC)
    declared = loads((copy / "pyproject.toml").read_text())["tool"]["uv"][
        "exclude-newer"
    ]
    (copy / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]", 'dependencies = ["aged"]', 'name = "probe"',
                'requires-python = ">=3.15"', 'version = "0"',
                "", "[tool.uv]",
                f'exclude-newer = "{declared}"\n'
            ]
        )
    )
    stand_in("bun", "exit 0")
    stood = stand_in("mise", "exit 0").log.parent
    path  = pathsep.join(
        entry for entry in environ["PATH"].split(pathsep) if Path(entry) != stood
    )
    stand_in("uv", f'exec {quote(which("uv", path=path))} "$@"')

    with serving({"1.0.0": now - 2 * window, "2.0.0": now - window / 2}) as address:
        monkeypatch.setenv("UV_DEFAULT_INDEX", f"{address}/simple")
        monkeypatch.setenv("UV_NO_CACHE", "1")
        result = executed(copy, "sync")

    assert result.returncode == 0, result.stderr
    assert 'name = "aged"\nversion = "1.0.0"' in (copy / "uv.lock").read_text()


def test_the_sync_fails_on_a_window_bun_cannot_be_given_before_locking(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn]
):
    """
    Asserts that `lock:sync` fails on an `exclude-newer` uv accepts but
    written other than as whole days, naming the value and the form it
    reads, before `mise` or `uv` locks anything.
    """
    manifest = copy / "pyproject.toml"
    manifest.write_text(
        sub(r'(?m)^(exclude-newer\s*=\s*)".*"$', r'\1"P1W"', manifest.read_text())
    )
    ran = [stand_in(program, "exit 0") for program in ("mise", "uv")]

    result = executed(copy, "sync")

    assert (
        result.returncode,
        result.stderr.strip(),
        [*ran[0].commands, *ran[1].commands]
    ) == (
        1,
        "[tool.uv] sets exclude-newer to 'P1W', which lock:sync passes Bun only when "
        "written as whole days, P<days>D",
        []
    )
