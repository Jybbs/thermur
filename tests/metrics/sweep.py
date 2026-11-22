"""
Pins `Sweep`, meaning the flights it holds by size beside each size's
fixed-speed control, the run length its pilot flight sets, and the slope of
each statistic against the size.
"""

from common.drones import short
from numpy         import array, nan
from numpy.testing import assert_allclose
from pytest        import MonkeyPatch, mark
from types         import SimpleNamespace
from xarray        import DataTree, Dataset

from thermur.controller.schemas import Composite
from thermur.metrics.flight     import Flight
from thermur.metrics.sweep      import CONTROL, Sweep
from thermur.pipeline.settings  import Settings


@mark.slow
def test_a_sweep_flies_each_size_beside_its_fixed_speed_control():
    """
    Asserts that a sweep of two sizes holds each size's statistics under the
    size, beside its control holding every drone at the cruise speed, each
    run no longer than the settling plus `longest_s`.
    """
    plan  = short()
    sweep = Sweep(
        longest_s = 1.0,
        pilot_s   = 2.0,
        seeds     = 2,
        sizes     = (9, 12)
    )

    tree = sweep.flown(Composite.from_settings(Settings(plan=plan)), plan)

    assert set(tree.children) == {"12", "9"}
    for size in sweep.sizes:
        node = tree[str(size)]
        assert node.attrs["drones"] == size
        assert plan.settled_s < node.attrs["duration_s"] <= plan.settled_s + 1.0
        assert node.sizes["seed"] == 2
        assert_allclose(node[CONTROL]["group_speed_m_s"], plan.cruise_m_s, rtol=1e-3)


def test_a_pilot_reading_no_relaxation_still_flies_the_pilot_s_length(
    monkeypatch: MonkeyPatch
):
    """
    Asserts that a size whose pilot integrates both relaxation times to zero
    flies `pilot_s` past the settling rather than the settling alone.
    """
    plan = short()
    monkeypatch.setattr(
        Flight,
        "flown",
        lambda air, composite, run: SimpleNamespace(
            autocorrelation_s = array([0.0]),
            rearrangement_s   = array([0.0]),
            still_statistics  = Dataset()
        )
    )

    tree = Sweep(pilot_s=3.0, sizes=(9,)).flown(
        Composite.from_settings(Settings(plan=plan)),
        plan
    )

    assert tree["9"].attrs["duration_s"] == plan.settled_s + 3.0


def test_a_statistic_growing_as_a_power_of_the_size_reads_that_power():
    """
    Asserts that a statistic growing as the square root of the size reads a
    slope of one half with no error, leaving out a seed it was not measured
    on, while a statistic negative on any seed reads no slope.
    """
    tree = DataTree.from_dict(
        {
            str(size): Dataset(
                {
                    "grown"  : ("seed", array([size**0.5, size**0.5, nan])),
                    "signed" : ("seed", array([1.0, -1.0, 1.0]))
                },
                attrs = {"drones": size}
            )
            for size in (10, 100, 1000)
        }
    )

    slopes = Sweep.slopes(tree)

    assert list(slopes.statistic.values) == ["grown"]
    assert_allclose(slopes.slope, [0.5])
    assert_allclose(slopes.error, [0.0], atol=1e-12)


def test_a_tree_reads_its_sizes_in_ascending_order_by_flight():
    """
    Asserts that `sized` reads a tree holding its sizes largest first in
    ascending order, each the composite's statistics by default and the
    control's under `CONTROL`.
    """
    tree = DataTree.from_dict(
        {
            path: Dataset({"read": ("seed", array([value]))}, attrs={"drones": size})
            for size in (100, 10)
            for path, value in ((str(size), size), (f"{size}/{CONTROL}", -size))
        }
    )

    assert {size: table.read.item() for size, table in Sweep.sized(tree).items()} == {
        10  : 10.0,
        100 : 100.0
    }
    assert list(Sweep.sized(tree, CONTROL)) == [10, 100]
    assert Sweep.sized(tree, CONTROL)[10].read.item() == -10.0
