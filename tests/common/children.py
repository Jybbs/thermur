"""
Holds `runs`, which a test starting a child process calls on each module
the child runs, since pytest-testmon traces only the test's own process, and
`spawned`, which runs a script in a child and calls `runs` on the `thermur`
modules that child loaded.
"""

from importlib.util import find_spec
from runpy          import run_path
from subprocess     import run
from sys            import executable


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
            f"{script}; import sys; print(*sorted(name for name, module in "
            "sys.modules.items() if name.startswith('thermur') "
            "and getattr(module, '__file__', None)))"
        ],
        capture_output = True,
        check          = True,
        text           = True
    )
    *printed, loaded = started.stdout.splitlines()
    runs(*loaded.split())

    return printed
