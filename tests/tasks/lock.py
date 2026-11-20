"""
Pins the path `lock:check` takes for each answer `uv`, `mise`, and `bun`
give it, each case running a copy of the task beside a lockfile of its
own, the lockfile beside each task script declaring its own dependencies
that every `lock` task checks, audits, or writes, and what the real mise
answers for the committed lockfile online and for one missing a platform
entry offline, where `lock:check` fails on a `mise lock` exiting nonzero
or counting a skipped platform and on a lockfile in which a tool holds no
entry for a platform another tool carries. It also pins which `AUDIT_IGNORE`
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

CHECKED    = "mise lock    ✓ 72 targets checked"
FAILED     = "mise ERROR error sending request"
LYCHEE     = '[tools."aqua:lycheeverse/lychee"."platforms.linux-x64"]'
MICROMATCH = "GHSA-952p-6rrq-rcjv"
MISSING = "No lockfile URL found for uv@0.12.22 on platform macos-arm64 (--locked mode)"
PICOMATCH = "GHSA-fc5q-39xc-vjpf"
SUMMARY   = "✓ Updated 14 platform entries ({skipped} skipped)"


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
        ".mise/config.toml", ".mise/lib", ".mise/mise.lock",
        ".mise/tasks/lock", "pyproject.toml"
    ):
        (pytestconfig.rootpath / path).copy(tmp_path / path, preserve_metadata=True)

    monkeypatch.delenv("AUDIT_IGNORE", raising=False)
    monkeypatch.setenv("MISE_PROJECT_ROOT", str(tmp_path))
    stand_in("bun", "exit 0")
    stand_in("gh", "exit 0")
    stand_in("uv", "exit 0")
    return tmp_path


@mark.parametrize(
    "summary",
    [
        param(SUMMARY.format(skipped=0), id="no-skipped-platform"),
        param(CHECKED, id="the-targets-checked")
    ]
)
def test_a_current_lock_passes(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn],
    summary  : str
):
    """
    Asserts that the task exits `0` where `uv.lock` is current, `mise lock`
    writes every platform and changes nothing, reporting no skipped platform
    or only the targets it checked, and the dry run installs every tool the
    project pins under the project's lock scope alone, while the task prints
    nothing and leaves `.mise/mise.lock` the file it found.
    """
    lock = copy / ".mise/mise.lock"
    held = (lock.read_bytes(), lock.stat().st_ino)
    stand_in(
        "mise",
        f'[ "$1" = lock ] && echo "{summary}"\n'
        'echo "$MISE_LOCKED_SCOPES $*" >> install\n'
        'echo "mise uv@0.12.22 would install"'
    )

    result = executed(copy)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert (lock.read_bytes(), lock.stat().st_ino) == held
    assert (copy / "install").read_text().splitlines()[-1] == (
        "project install --dry-run --force --locked"
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


def test_a_rewritten_lock_fails_and_is_restored(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn]
):
    """
    Asserts that a `mise lock` rewriting `.mise/mise.lock` fails the task,
    which prints the difference and leaves the file as it found it, byte for
    byte and mode for mode.
    """
    lock = copy / ".mise/mise.lock"
    held = (lock.read_bytes(), lock.stat().st_mode)
    stand_in(
        "mise",
        'echo "drifted = true" >> .mise/mise.lock\n'
        "chmod 600 .mise/mise.lock\n"
        f'echo "{SUMMARY.format(skipped=0)}"'
    )

    result = executed(copy)

    assert (result.returncode, "+drifted = true" in result.stderr) == (1, True)
    assert (lock.read_bytes(), lock.stat().st_mode) == held


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
    stand_in("mise", f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}"\nexit 0')

    result = executed(copy)

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
    copy     : Path,
    mise     : StandIn,
    reported : str
):
    """
    Asserts that a `mise lock` counting a skipped platform, which exits `0`,
    a `mise lock` exiting nonzero, and a dry run finding no lockfile entry
    for a pinned tool on the running platform each fail the task, which
    prints what mise reported.
    """
    result = executed(copy)

    assert (result.returncode, result.stderr.strip()) == (1, reported)


def test_a_tool_missing_a_platform_another_carries_fails_naming_both(
    copy     : Path,
    stand_in : Callable[[str, str], StandIn]
):
    """
    Asserts that a lockfile in which one tool holds no entry for a platform
    another tool's entries carry fails the task, which names the tool
    and the platform, even where `mise lock` exits `0`, counts no skipped
    platform, and leaves the file as it found it.
    """
    lock = copy / ".mise/mise.lock"
    lock.write_text(sub(rf"{escape(LYCHEE)}\n(?:.+\n)+\n?", "", lock.read_text()))
    stand_in("mise", f'echo "{CHECKED}"')

    result = executed(copy)

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
    copy        : Path,
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
    lock    = copy / ".mise/mise.lock"
    trimmed = sub(rf"{escape(table)}\n(?:.+\n)+\n?", "", lock.read_text())
    lock.write_text(trimmed)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("MISE_HTTP_RETRIES", "0")
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(copy))

    assert (executed(copy).returncode, lock.read_text()) == (1, trimmed)


@mark.network
def test_the_committed_lock_passes_the_check_online(
    copy        : Path,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the real `mise`, locking every platform `.mise/mise.lock`
    holds and running the dry run, passes the task on the committed
    lockfile, printing nothing. The release `min_version` names fetches each
    platform entry over the network to lock it.
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
    stand_in("mise", f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}"\nexit 0')

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
