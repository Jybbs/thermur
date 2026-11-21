"""
Pins what `repo:bump` moves for each flag it takes, meaning the release
`uv version --bump` writes into `[project].version`, the `thermur` entry of
`uv.lock` and the `version` of `CITATION.cff` the task writes after it, and
the status it exits with where both flags are given.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import (
    CaptureFixture, Config, MonkeyPatch, fixture, mark, param, raises
)
from re    import sub
from types import ModuleType

LOCK = """
[[package]]
name = "filelock"
version = "0.1.0"

[[package]]
name = "thermur"
version = "{version}"
"""


@fixture
def bump(
    load_task    : Callable[[str], ModuleType],
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    tmp_path     : Path,
    uv           : StandIn
) -> ModuleType:
    """
    Enters a folder holding the worktree's `pyproject.toml` at `0.1.0`, a
    `CITATION.cff` and a `uv.lock` naming `thermur` and one other package at
    that release, behind the stand-in `uv`.

    Returns:
        The task script, imported as a module.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        sub(
            r'(?m)^(version\s*=\s*)".*"',
            r'\1"0.1.0"',
            (pytestconfig.rootpath / "pyproject.toml").read_text(),
            count = 1
        )
    )
    (tmp_path / "uv.lock").write_text(LOCK.format(version="0.1.0"))
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ndate-released: 2026-10-02\nversion: 0.1.0\n"
    )
    return load_task(".mise/tasks/repo/bump.py")


@fixture
def uv(stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `uv` on the path that prints the release `0.1.0` moves
    to under the part its `--bump` names.
    """
    return stand_in(
        "uv",
        'case "$3" in\n'
        "  major) echo 1.0.0 ;;\n"
        "  minor) echo 0.2.0 ;;\n"
        "  patch) echo 0.1.1 ;;\n"
        "esac"
    )


@mark.parametrize(
    ("flags", "part", "version"),
    [
        param({}, "minor", "0.2.0", id="minor"),
        param({"major": "false", "patch": "false"}, "minor", "0.2.0", id="minor-unset"),
        param({"patch": "true"}, "patch", "0.1.1", id="patch"),
        param({"major": "true"}, "major", "1.0.0", id="major")
    ]
)
def test_the_bump_moves_each_file_to_the_release_its_flag_names(
    bump        : ModuleType,
    capsys      : CaptureFixture[str],
    flags       : dict[str, str],
    monkeypatch : MonkeyPatch,
    part        : str,
    tmp_path    : Path,
    uv          : StandIn,
    version     : str
):
    """
    Asserts that the bump asks `uv version` for the next minor release by
    default or where mise passes each flag as `false`, or the patch or major
    one its flag names, writing the manifest through uv without a relock,
    and writes the release uv prints into the `thermur` entry of `uv.lock`
    and the `version` of `CITATION.cff`, leaving every other pin and key as
    it was.
    """
    for flag, given in flags.items():
        monkeypatch.setenv(f"usage_{flag}", given)

    bump.main()

    assert uv.commands == [
        f"uv version --bump {part} --frozen --short --dry-run",
        f"uv version --bump {part} --frozen --short"
    ]
    assert (tmp_path / "uv.lock").read_text() == LOCK.format(version=version)
    assert (tmp_path / "CITATION.cff").read_text() == (
        f"cff-version: 1.2.0\ndate-released: 2026-10-02\nversion: {version}\n"
    )
    assert capsys.readouterr().out == f"{version}\n"


def test_a_citation_at_another_release_fails_before_the_lock_moves(
    bump     : ModuleType,
    tmp_path : Path,
    uv       : StandIn
):
    """
    Asserts that a `CITATION.cff` naming a release other than the manifest's
    fails the bump on that file before `uv` writes the manifest, leaving the
    `thermur` entry of `uv.lock` where it stood.
    """
    (tmp_path / "CITATION.cff").write_text("version: 0.0.9\n")

    with raises(ValueError, match="`CITATION.cff` names no `0.1.0`"):
        bump.main()

    assert (tmp_path / "uv.lock").read_text() == LOCK.format(version="0.1.0")
    assert uv.commands == ["uv version --bump minor --frozen --short --dry-run"]


def test_two_flags_exit_before_any_file_moves(
    bump        : ModuleType,
    monkeypatch : MonkeyPatch,
    uv          : StandIn
):
    """
    Asserts that `--patch` beside `--major` exits with status `2` before
    `uv` runs.
    """
    monkeypatch.setenv("usage_major", "true")
    monkeypatch.setenv("usage_patch", "true")

    with raises(SystemExit) as exited:
        bump.main()

    assert (exited.value.code, uv.commands) == (2, [])
