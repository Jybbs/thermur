"""
Pins that `lock:sync`, running the real `uv` against a stand-in registry,
leaves out of `uv.lock` a release newer than the window `[tool.uv]`
declares, and fails before locking on a window written other than as whole
days.
"""

from collections.abc  import Callable
from common.registry  import serving
from common.stand_ins import StandIn
from common.tasks     import run_lock
from datetime         import UTC, datetime
from os               import environ, pathsep
from pathlib          import Path
from pytest           import MonkeyPatch, mark
from re               import sub
from shlex            import quote
from shutil           import which
from tomllib          import loads

from thermur.repo.checkout import Checkout


@mark.enable_socket
@mark.allow_hosts(["127.0.0.1"])
def test_the_sync_leaves_a_release_newer_than_the_window_out_of_uv_lock(
    lock_copy   : Path,
    monkeypatch : MonkeyPatch,
    stand_in    : Callable[[str, str], StandIn]
):
    """
    Asserts that `lock:sync`, resolving the package `aged` through the real
    `uv` against a registry publishing `1.0.0` twice the window `[tool.uv]`
    declares ago and `2.0.0` half of it ago, locks `1.0.0` in `uv.lock`,
    reading the window the worktree's `pyproject.toml` declares.

    The manifest the case writes sets `offline` to `false`, which outranks
    the same key in a system `uv.toml` such as `/etc/uv/uv.toml`.
    """
    window   = Checkout(root=lock_copy).manifest.window
    now      = datetime.now(UTC)
    declared = loads((lock_copy / "pyproject.toml").read_text())["tool"]["uv"][
        "exclude-newer"
    ]
    (lock_copy / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]", 'dependencies = ["aged"]', 'name = "probe"',
                'requires-python = ">=3.15"', 'version = "0"',
                "", "[tool.uv]",
                f'exclude-newer = "{declared}"',
                "offline = false\n"
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
        result = run_lock(lock_copy, "sync")

    assert result.returncode == 0, result.stderr
    assert 'name = "aged"\nversion = "1.0.0"' in (lock_copy / "uv.lock").read_text()


def test_the_sync_fails_on_a_window_bun_cannot_be_given_before_locking(
    lock_copy : Path,
    stand_in  : Callable[[str, str], StandIn]
):
    """
    Asserts that `lock:sync` fails on an `exclude-newer` uv accepts but
    written other than as whole days, naming the value and the form it
    reads, before `mise` or `uv` locks anything.
    """
    manifest = lock_copy / "pyproject.toml"
    manifest.write_text(
        sub(r'(?m)^(exclude-newer\s*=\s*)".*"$', r'\1"P1W"', manifest.read_text())
    )
    ran = [stand_in(program, "exit 0") for program in ("mise", "uv")]

    result = run_lock(lock_copy, "sync")

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
