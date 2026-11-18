"""
Holds `runs`, which a test starting a child process calls on each module the
child runs, since pytest-testmon traces only the test's own process.
"""

from importlib.util import find_spec
from runpy          import run_path


def runs(*modules: str):
    """
    Runs each of `modules` afresh in this process, leaving `sys.modules` as
    it stands, so pytest-testmon records the test as reading each module's
    top-level block. A change inside a function only the child calls still
    selects no test.
    """
    for module in modules:
        run_path(find_spec(module).origin, run_name=module)
