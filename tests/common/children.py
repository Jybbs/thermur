"""
Holds `runs`, which a test starting a child process calls on each module the
child runs, since pytest-testmon traces only the test's own process. Beside
it, `spawned` runs a script in a child and calls `runs` on each `thermur`
module that child loaded, and `nested` runs a second pytest in a child.
"""

from collections.abc import Mapping, Sequence
from importlib.util  import find_spec
from os              import environ
from pathlib         import Path
from pytest          import TempPathFactory
from runpy           import run_path
from subprocess      import CompletedProcess, PIPE, run
from sys             import executable


def nested(
    command          : Sequence[str | Path],
    tmp_path_factory : TempPathFactory,
    *,
    check            : bool              = False,
    cwd              : Path | None       = None,
    env              : Mapping[str, str] = environ
) -> CompletedProcess[str]:
    """
    Runs `command`, which starts pytest, in a child under `env` with
    `PYTHONDONTWRITEBYTECODE` set, so no module of a suite under the
    temporary root reaches the bytecode cache `PYTHONPYCACHEPREFIX` names,
    and with a `--basetemp` in a fresh folder of `tmp_path_factory`, so the
    child opens no numbered folder of its own in the machine-wide root.

    Returns:
        The finished run, its output captured as text.
    """
    return run(
        [*command, f"--basetemp={tmp_path_factory.mktemp('inner')}"],
        capture_output = True,
        check          = check,
        cwd            = cwd,
        env            = {**env, "PYTHONDONTWRITEBYTECODE": "1"},
        text           = True
    )


def runs(*modules: str):
    """
    Runs the top-level block of each of `modules` in this process, in a
    namespace of its own, so pytest-testmon records the test as reading that
    block. The entry `sys.modules` holds under each name stays as it was,
    whereas each parent package `find_spec` imports and each module a block
    imports as it runs stay loaded. A change inside a function only the
    child calls still selects no test.
    """
    for module in modules:
        run_path(find_spec(module).origin, run_name=module)


def spawned(script: str) -> list[str]:
    """
    Runs `script` in a fresh interpreter of the environment, ending on a
    print of each `thermur` module in its `sys.modules` loaded from a file,
    then calls `runs` on those modules.

    Returns:
        The lines `script` printed.
    """
    started = run(
        [
            executable, "-c",
            f"{script}\nimport sys\nprint(*sorted(name for name, module in "
            "sys.modules.items() if name.startswith('thermur') "
            "and getattr(module, '__file__', None)))"
        ],
        check  = True,
        stdout = PIPE,
        text   = True
    )
    *printed, loaded = started.stdout.splitlines()
    runs(*loaded.split())

    return printed
