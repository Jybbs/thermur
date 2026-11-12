"""
Pins the flight's steps, meaning the reference flight the settings describe,
its frames and statistics as tables, and one trial step per open slot.
"""

from common.drones import short, still_air
from contextlib    import closing
from gc            import collect
from numpy.testing import assert_array_equal
from threading     import Thread

from thermur.controller.components import Elastic
from thermur.controller.schemas    import Composite
from thermur.environment.schemas   import Crop, Grid, ParametricThreat
from thermur.metrics               import steps
from thermur.metrics.flight        import Flight
from thermur.metrics.trials        import Measure, Trial
from thermur.pipeline.graph        import Graph
from thermur.pipeline.settings     import Settings
from thermur.repo.checkout         import Checkout


def test_a_run_leaves_no_connection_for_a_collection_on_another_thread(clone: Checkout):
    """
    Asserts that a computed graph leaves no SQLite connection open, so
    a collection on a thread other than the one that ran it, such as the
    thread a trial step's host callback runs on, frees the graph without the
    `ResourceWarning` an unclosed connection raises.
    """
    Graph(checkout=clone).compute("composite")
    collector = Thread(target=collect)
    collector.start()
    collector.join()


def test_a_trial_step_flies_its_trial_against_the_settings_composite():
    """
    Asserts that a trial step flies the trial it is given under the
    composite and the plan the settings hold.
    """
    trial = Trial(
        candidates = {"elastic": Elastic()},
        conditions = {"small": {}},
        duration_s = 2.0,
        measures   = (Measure(statistic="spacing_m"),),
        slot       = "cohesion"
    )
    settings = Settings(plan=short())

    flown = steps.flown_trial(
        still_air(),
        Composite.from_settings(settings),
        "digest",
        settings,
        trial
    )

    assert list(flown.candidate.values) == ["reference", "elastic"]
    assert flown.spacing_m.dims == ("candidate", "condition", "seed")


def test_each_open_slot_is_tried_by_a_step_of_its_own(clone: Checkout):
    """
    Asserts that the graph declares one trial step for each listed trial.
    """
    with closing(Graph(checkout=clone)) as graph:
        nodes = graph.driver.graph.nodes

    assert {trial.name for trial in Trial.listed()} <= set(nodes)


def test_the_graph_flies_the_plan_the_settings_hold(clone: Checkout):
    """
    Asserts that the `flight` step flies the plan the settings hold under
    their composite, and that `frames` and `statistics` lay that flight out.
    """
    settings = Settings(
        crop   = Crop(floor_m=20.0, levels=5, spacing_m=20.0),
        plan   = short(center_m=(160.0, 40.0, 60.0)),
        threat = ParametricThreat(
            edge_m = 10.0,
            grid   = Grid(
                columns   = 8,
                levels    = 10,
                rows      = 2,
                snapshots = 2
            )
        )
    )

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
