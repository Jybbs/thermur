"""
Holds `runs`, which a test starting a child process calls on each module the
child runs, since pytest-testmon traces only the test's own process. Beside
it, `spawned` runs a script in a child and calls `runs` on each `thermur`
module that child loaded, and `nested` runs a second pytest in a child.
"""

from collections.abc import Sequence
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
    check            : bool        = False,
    cwd              : Path | None = None,
    **variables      : str
) -> CompletedProcess[str]:
    """
    Runs `command`, which starts pytest, in a child with `variables` and
    `PYTHONDONTWRITEBYTECODE` set over this process's environment and a
    `--basetemp` in a fresh folder of `tmp_path_factory`, so the child
    writes no bytecode for its suite and opens no folder in the machine-wide
    root.

    Returns:
        The finished run, its output captured as text.
    """
    return run(
        [*command, f"--basetemp={tmp_path_factory.mktemp('inner')}"],
        capture_output = True,
        check          = check,
        cwd            = cwd,
        env            = {**environ, **variables, "PYTHONDONTWRITEBYTECODE": "1"},
        text           = True
    )


def runs(*modules: str):
    """
    Runs the top-level block of each of `modules` in this process, in a
    namespace of its own, so pytest-testmon records the test as reading
    that block. The entry `sys.modules` holds under each name stays as it
    was, whereas each parent package `find_spec` imports and each module a
    block imports as it runs stay loaded. A change inside a function only
    the child calls selects no test unless the top-level block runs a line
    in the span pytest-testmon checksums as that function's block. The span
    starts on the line of the function's first statement and ends on the
    line before the next statement in the body holding the function. Where
    the function ends that body, the span ends before whatever statement
    or `except` clause follows the statement, `except` clause, or `case`
    clause holding the body, stepping outward through each enclosing one
    that nothing follows, and ends on the function's last line once that
    stepping reaches a `case` clause another follows. Each line of the span
    the top-level block runs is a line of the `def` statement the function's
    body shares, a decorator of the statement after the span, since `ast`
    starts a decorated statement on its `def` or `class` line, or a line of
    a later branch of an enclosing statement, such as an `elif`, an `else`,
    an `except`, or a `finally`. Adding or removing a `def` or `async def`
    inside the function selects the test regardless, since pytest-testmon
    checksums each span with its number in the order it finishes them, and
    finishes the span of the whole module, which the test reads, last.
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
