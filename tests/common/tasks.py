"""
Runs a copy of a `lock` task or a `site` task the way mise runs it, which
the `lock` tests in each script's module and the `site` tests share.
"""

from pathlib    import Path
from pytest     import Config
from subprocess import CompletedProcess, run


def run_lock(copy: Path, task: str = "check") -> CompletedProcess:
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


def run_site(cwd: Path, pytestconfig: Config, task: str) -> CompletedProcess:
    """
    Runs the `site` task named through its own shebang from `cwd`, with
    `--marker` as an argument a reader adds, capturing what it prints.
    """
    return run(
        [pytestconfig.rootpath / ".mise/tasks/site" / task, "--marker"],
        capture_output = True,
        cwd            = cwd,
        text           = True
    )
