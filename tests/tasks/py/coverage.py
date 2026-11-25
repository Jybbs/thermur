"""
Pins the reports `py:coverage` writes and the status it exits with.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pytest           import Config, mark, param
from subprocess       import run


@mark.parametrize(
    ("measured", "reported", "status"),
    [
        param(0, 0, 0, id="both-pass"),
        param(1, 0, 1, id="a-test-fails"),
        param(0, 2, 2, id="below-fail-under"),
        param(1, 2, 2, id="both-fail")
    ]
)
def test_the_coverage_reports_whatever_the_suite_returns(
    measured     : int,
    pytestconfig : Config,
    reported     : int,
    stand_in     : Callable[[str, str], StandIn],
    status       : int
):
    """
    Asserts that `py:coverage` writes the HTML report at any total and
    runs the terminal report, which fails below `fail_under`, whether or
    not a test failed, exiting with the report's status where it fails and
    otherwise with the status of the suite.
    """
    stand_in("pytest", f"exit {measured}")
    coverage = stand_in(
        "coverage",
        f'[ "$*" = "html --fail-under=0" ] || exit {reported}'
    )
    ran = run(
        [pytestconfig.rootpath / ".mise/tasks/py/coverage"],
        capture_output = True,
        cwd            = pytestconfig.rootpath
    )

    assert (ran.returncode, coverage.commands) == (
        status,
        ["coverage html --fail-under=0", "coverage report"]
    )
