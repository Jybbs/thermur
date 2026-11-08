"""
Pins that the `data:fetch` task runs the command fetching the sample, and
that `data:browse` opens the experiment viewer over the recorded runs.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pytest           import Config
from subprocess       import run


def test_browsing_opens_the_viewer_over_the_recorded_runs(
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that the `data:browse` task runs `h_experiments` over
    `data/runs`, passing each argument it is given before the folder.
    """
    root = pytestconfig.rootpath
    stand_in("h_experiments", 'printf "%s\\n" "$@"')

    printed = run(
        [root / ".mise/tasks/data/browse", "--port", "8124"],
        capture_output = True,
        check          = True,
        cwd            = root,
        text           = True
    ).stdout.splitlines()

    assert printed == ["--port", "8124", "data/runs"]


def test_fetching_runs_the_command_fetching_the_sample(
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that the `data:fetch` task runs `thermur fetch sample`, passing
    the command nothing more.
    """
    root = pytestconfig.rootpath
    stand_in("thermur", 'printf "%s\\n" "$@"')

    printed = run(
        [root / ".mise/tasks/data/fetch"],
        capture_output = True,
        check          = True,
        cwd            = root,
        text           = True
    ).stdout.splitlines()

    assert printed == ["fetch", "sample"]
