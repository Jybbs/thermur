"""
Pins the flight's steps, meaning the reference flight the settings describe,
its frames and statistics as tables, and one trial step per open slot.
"""

from common.declarations import DECLARED
from common.drones       import short, small, still_air
from numpy.testing       import assert_array_equal
from pytest              import MonkeyPatch, mark
from xarray              import Dataset

from thermur.controller.schemas import Composite
from thermur.metrics            import steps
from thermur.metrics.flight     import Flight
from thermur.metrics.trials     import Trial
from thermur.pipeline.graph     import Graph
from thermur.pipeline.settings  import Settings
from thermur.repo.checkout      import Checkout

pytestmark = mark.reads(*DECLARED, "src/**")


def test_a_trial_step_flies_its_trial_against_the_settings_composite(
    monkeypatch: MonkeyPatch
):
    """
    Asserts that a trial step flies the trial it is given through the
    air and under the composite it reads and the plan the settings hold,
    returning the table the trial returns.
    """
    trial = Trial(
        candidates = {},
        conditions = {},
        measures   = (),
        slot       = "cohesion"
    )
    settings  = Settings(plan=short())
    air       = still_air()
    composite = Composite.from_settings(settings)
    received  = []
    table     = Dataset()
    monkeypatch.setattr(
        Trial,
        "flown",
        lambda *passed: received.append(passed) or table
    )

    flown = steps.flown_trial(
        air,
        composite,
        "digest",
        settings,
        trial
    )

    assert flown is table
    assert received == [(trial, air, composite, settings.plan)]


def test_each_open_slot_is_tried_by_a_step_of_its_own(clone: Checkout):
    """
    Asserts that the graph declares one trial step for each listed trial.
    """
    nodes = Graph(checkout=clone).driver.graph.nodes

    assert {trial.name for trial in Trial.listed()} <= set(nodes)


def test_the_graph_flies_the_plan_the_settings_hold(clone: Checkout):
    """
    Asserts that the `flight` step flies the plan the settings hold under
    their composite, and that `frames` and `statistics` lay that flight out.
    """
    settings = small()

    computed = Graph(checkout=clone, settings=settings).compute(
        "flight",
        "frames",
        "statistics"
    )

    assert isinstance(computed["flight"], Flight)
    assert computed["flight"].plan == settings.plan
    assert_array_equal(
        computed["frames"].positions,
        computed["flight"].frames.positions
    )
    assert computed["statistics"].identical(computed["flight"].statistics)
