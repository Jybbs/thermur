"""
Pins that every `lock` task reaches the lockfile beside each task script
declaring its own dependencies, beside `uv.lock` and `docs/bun.lock`.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from common.tasks     import SUMMARY, run_lock
from pathlib          import Path
from pytest           import mark, param

from thermur.repo.checkout import Checkout


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
    lock_copy   : Path,
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
    age    = int(Checkout(root=lock_copy).manifest.window.total_seconds())
    script = ".mise/tasks/gha/declared.py"
    (lock_copy / ".mise/tasks/gha").mkdir()
    (
        lock_copy / script
    ).write_text('# /// script\n# dependencies = ["minijinja"]\n# ///\n')
    (lock_copy / ".mise/tasks/gha/plain.py").write_text('"""\nDeclares nothing.\n"""\n')
    programs = {program: stand_in(program, "exit 0") for program in ("bun", "uv")}
    stand_in("mise", f'[ "$1" = lock ] && echo "{SUMMARY.format(skipped=0)}"\nexit 0')

    assert run_lock(lock_copy, task).returncode == 0
    assert programs["uv"].commands == [
        f"uv {command.format(script=script)}" for command in ran
    ]
    assert programs["bun"].commands == [f"bun {bun_command.format(age=age)}"]
