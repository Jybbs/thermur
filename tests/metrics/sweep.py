"""
Pins `Sweep`, meaning the flights it holds by size beside each size's
fixed-speed control, the run length its pilot flight sets, and the slope of
each statistic against the size.
"""

from common.drones import short
from numpy         import array, nan
from numpy.testing import assert_allclose
from pytest        import mark
from xarray        import DataTree, Dataset

from thermur.controller.schemas import Composite
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
