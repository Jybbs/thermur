"""
Pins `measured`, meaning the summary by rule and condition it reads out of
the trial tables a run saved.
"""

from numpy   import array
from pathlib import Path
from xarray  import Dataset

from thermur.docs.measured  import measured
from thermur.metrics.trials import REFERENCE


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
