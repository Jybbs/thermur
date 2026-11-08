"""
Pins `measured`, meaning the given value and the summary by rule and
condition it reads out of the trial tables a run saved.
"""

from numpy   import array
from pathlib import Path
from xarray  import Dataset

from thermur.docs.measured  import measured
from thermur.metrics.trials import REFERENCE


def test_the_trials_read_by_slot_given_value_rule_and_condition(tmp_path: Path):
    """
    Asserts that each slot reads its conditions' launch spacing or turn,
    that a condition its statistic never measured drops out, and that each
    rule's mean and standard error across seeds sit under the condition's
    name.
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

    assert trials["coupling"]["given"] == {
        "dense"  : 10.0,
        "middle" : 15.0,
        "sparse" : 25.0
    }
    assert trials["heading"]["given"] == {"large": 90.0, "small": 20.0}
    assert trials["coupling"]["measured"]["other"]["sparse"] == {
        "error" : 0.0,
        "mean"  : 0.2
    }
    assert trials["heading"]["measured"][REFERENCE] == {
        "large" : {"error": 0.0707, "mean": 0.6},
        "small" : {"error": 0.0, "mean": 0.4}
    }
