"""
Pins `measured`, meaning the summary by rule and condition it reads out of
the trial tables a run saved, and `swept`, the summary by statistic, flight,
and size it reads out of the sweep a run saved.
"""

from numpy   import array, nan
from pathlib import Path
from xarray  import DataTree, Dataset

from thermur.docs.measured  import measured, swept
from thermur.metrics.sweep  import CONTROL
from thermur.metrics.trials import REFERENCE


def test_the_sweep_reads_each_statistic_by_flight_and_size(tmp_path: Path):
    """
    Asserts that the sweep's means and standard errors sit by statistic,
    flight, and size, a mean no seed measured written as `None`, beside each
    size's run, the slopes against the size, and the rank correlation of the
    group speed with the size.
    """
    DataTree.from_dict(
        {
            path: Dataset(
                {
                    "group_speed_m_s" : ("seed", array([size, size + 1.0])),
                    "wave_speed_m_s"  : ("seed", array([nan, nan]))
                },
                attrs = {
                    "autocorrelation_s" : 2.0,
                    "drones"            : size,
                    "duration_s"        : 30.0,
                    "rearrangement_s"   : 3.0
                } if path == str(size) else {}
            )
            for size in (10, 100)
            for path in (str(size), f"{size}/{CONTROL}")
        }
    ).to_netcdf(tmp_path / "sweep.nc")

    summary = swept(tmp_path)

    assert summary["sizes"] == [10, 100]
    assert summary["runs"]["100"] == {
        "autocorrelation_s" : 2.0,
        "duration_s"        : 30.0,
        "rearrangement_s"   : 3.0
    }
    assert summary["statistics"]["group_speed_m_s"]["flocking"][1] == {
        "error" : 0.3536,
        "mean"  : 100.5,
        "size"  : 100
    }
    assert summary["statistics"]["wave_speed_m_s"][CONTROL][0]["mean"] is None
    assert set(summary["slopes"]) == {"group_speed_m_s"}
    assert summary["speed_by_size"]["rho"] == 0.8944


def test_the_trials_read_by_slot_rule_and_condition(tmp_path: Path):
    """
    Asserts that a condition its statistic never measured drops out,
    that each rule's mean and standard error across seeds sit under the
    condition's name, and that nothing beside the rules is read.
    """
    for name, statistic, conditions in (
        ("coupling_trial", "polarization", ["dense", "middle", "sparse"]),
        ("heading_trial", "turn_reach", ["large", "small", "spontaneous"])
    ):
        values = array([[[0.5, 0.7], [0.4, 0.4], [float("nan")] * 2]] * 2)
        if name == "coupling_trial":
            values[:, 2] = 0.2
        Dataset(
            {statistic: (("candidate", "condition", "seed"), values)},
            coords = {"condition": conditions, "candidate": [REFERENCE, "other"]}
        ).to_netcdf(tmp_path / f"{name}.nc")

    trials = measured(tmp_path)

    assert {slot: set(rules) for slot, rules in trials.items()} == {
        "coupling" : {"other", REFERENCE},
        "heading"  : {"other", REFERENCE}
    }
    assert trials["coupling"]["other"]["sparse"] == {
        "error" : 0.0,
        "mean"  : 0.2
    }
    assert trials["heading"][REFERENCE] == {
        "large" : {"error": 0.0707, "mean": 0.6},
        "small" : {"error": 0.0, "mean": 0.4}
    }
