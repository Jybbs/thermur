"""
Pins the path `lock:check` takes for each answer `uv` and `mise` give it,
each case running a copy of the task beside a lockfile of its own, the
lockfile beside each task script declaring its own dependencies that every
`lock` task checks, audits, or writes, and the summary line the pinned mise
prints for a lock it writes in full, which `lock:check` reads to fail where
`mise lock` skips a platform, as it does offline.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import Config, FixtureRequest, MonkeyPatch, fixture, mark, param
from subprocess       import CompletedProcess, run
from tomllib          import loads

MISSING = "No lockfile URL found for uv@0.12.22 on platform macos-arm64 (--locked mode)"
SUMMARY = "✓ Updated 14 platform entries ({skipped} skipped)"


def checked(copy: Path) -> CompletedProcess:
    """
    Runs the copy of `lock:check` through its own shebang from the root of
    the copy, capturing what it prints.
    """
    return run(
        [copy / ".mise/tasks/lock/check"],
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
    and `.mise/mise.lock` into `tmp_path`, keeping each file's mode,
    names the copy as the root a task sources its library from, and puts a
    stand-in `uv` that reports `uv.lock` current ahead on the path.

    Returns:
        The copy's root.
    """
    (tmp_path / ".mise/tasks").mkdir(parents=True)
    for path in (
        ".mise/config.toml", ".mise/lib", ".mise/mise.lock", ".mise/tasks/lock"
    ):
        (pytestconfig.rootpath / path).copy(tmp_path / path, preserve_metadata=True)

    monkeypatch.setenv("MISE_PROJECT_ROOT", str(tmp_path))
    stand_in("uv", "exit 0")
    return tmp_path


def test_a_current_lock_passes(copy: Path, stand_in: Callable[[str, str], StandIn]):
    """
    Asserts that the task exits `0` where `uv.lock` is current, `mise lock`
    writes every platform and changes nothing, and the dry run installs
    every tool the project pins under the project's lock scope alone, while
    the task prints nothing and leaves `.mise/mise.lock` the file it found.
    """
    lock = copy / ".mise/mise.lock"
    held = (lock.read_bytes(), lock.stat().st_ino)
    stand_in(
        "mise",
        f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}"\n'
        'echo "$MISE_LOCKED_SCOPES $*" >> install\n'
        'echo "mise uv@0.12.22 would install"'
    )

    result = checked(copy)

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

    assert (checked(copy).returncode, (copy / "ran").exists()) == (1, False)


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

    result = checked(copy)

    assert (result.returncode, "+drifted = true" in result.stderr) == (1, True)
    assert (lock.read_bytes(), lock.stat().st_mode) == held


@fixture
def mise(request: FixtureRequest, stand_in: Callable[[str, str], StandIn]):
    """
    Writes a stand-in `mise` running the `sh` lines a case passes through
    `request.param`.
    """
    stand_in("mise", request.param)


@mark.parametrize(
    ("mise", "reported"),
    [
        param(
            f'echo "{SUMMARY.format(skipped=3)}"',
            SUMMARY.format(skipped=3),
            id = "skipped-platform"
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
    mise     : None,
    reported : str
):
    """
    Asserts that a `mise lock` skipping a platform, which exits `0`, and
    a dry run finding no lockfile entry for a pinned tool on the running
    platform each fail the task, which prints what mise reported.
    """
    result = checked(copy)

    assert (result.returncode, result.stderr.strip()) == (1, reported)


def test_an_offline_lock_fails_the_check(copy: Path, monkeypatch: MonkeyPatch):
    """
    Asserts that the real `mise lock`, reaching no release host through a
    proxy that refuses every connection, skips a platform and so fails the
    task, while exiting `0` itself.

    `MISE_HTTP_RETRIES` set to `0` makes each refused request fail at once
    rather than after mise's retries.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("MISE_HTTP_RETRIES", "0")
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(copy))

    result = checked(copy)

    assert result.returncode == 1
    assert "skipped)" in result.stderr
    assert "(0 skipped)" not in result.stderr


@mark.network
def test_mise_lock_writes_every_platform_online(
    copy        : Path,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the pinned mise, locking every platform `.mise/mise.lock`
    holds, prints the summary line `lock:check` reads, naming each platform
    entry and no skipped one.
    """
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(copy))
    entries = sum(
        key.startswith("platforms.")
        for versions in loads((copy / ".mise/mise.lock").read_text())["tools"].values()
        for entry in versions
        for key in entry
    )

    locked = run(["mise", "lock"], capture_output=True, cwd=copy, text=True)

    assert f"✓ Updated {entries} platform entries (0 skipped)" in (
        locked.stdout + locked.stderr
    ).splitlines()


@mark.parametrize(
    ("task", "ran"),
    [
        param("check", ["lock --check", "lock --check --script {script}"], id="check"),
        param(
            "scan",
            [
                "audit --preview-features audit-command",
                "audit --preview-features audit-command --script {script}"
            ],
            id = "scan"
        ),
        param(
            "sync",
            ["lock --no-locked", "lock --no-locked --script {script}"],
            id = "sync"
        )
    ]
)
def test_each_lock_task_reaches_the_lockfile_beside_a_script(
    copy     : Path,
    ran      : list[str],
    stand_in : Callable[[str, str], StandIn],
    task     : str
):
    """
    Asserts that each `lock` task runs its `uv` command over `uv.lock` and
    then over the lockfile beside each task script whose inline metadata
    opens a `script` block, leaving out a script declaring none.
    """
    script = ".mise/tasks/gha/declared.py"
    (copy / ".mise/tasks/gha").mkdir()
    (copy / script).write_text('# /// script\n# dependencies = ["minijinja"]\n# ///\n')
    (copy / ".mise/tasks/gha/plain.py").write_text('"""\nDeclares nothing.\n"""\n')
    uv = stand_in("uv", "exit 0")
    stand_in("mise", f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}"\nexit 0')

    finished = run(
        [copy / ".mise/tasks/lock" / task],
        capture_output = True,
        cwd            = copy,
        text           = True
    )

    assert finished.returncode == 0
    assert uv.commands == [f"uv {command.format(script=script)}" for command in ran]
