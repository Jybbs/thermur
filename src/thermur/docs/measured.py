"""
Holds `measured`, which reads the means and standard errors the charts on
The Interaction Rules draw out of the coupling and heading trials a run
saved, and `MEASURED`, the file the `docs:measure` task writes them into.
"""

from pathlib import Path
from xarray  import Dataset, open_dataset

from thermur.metrics.trials import Trial, summarized

MEASURED = Path("docs/.vitepress/lib/measured/trials.json")
"""
The file the charts read the trials' means from, relative to the clone's
root.
"""

DIGITS = 4
"""
The decimal places each mean and standard error keeps.
"""


def measured(run: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """
    Reads the polarization each coupling rule holds under each condition
    and the share of each turn each heading rule carries to the farthest
    drones, out of the trial tables the run at `run` saved, leaving what
    each condition sets to the file `mise render` writes from `Trial`.

    Args:
        run: The run's folder under `data/runs/`, holding the
             `coupling_trial.nc` and `heading_trial.nc` that `thermur fly
             coupling_trial heading_trial` saves.

    Returns:
        Each rule's mean and standard error across seeds by condition,
        under the name of the trial's slot.
    """
    trials = {trial.slot: trial for trial in Trial.listed()}
    return {
        slot: tabled(
            summary = summarized(
                (
                    open_dataset(run / f"{trials[slot].name}.nc")[statistic]
                    .dropna("condition", how="all")
                    .astype("float64")
                )
            ).round(DIGITS)
        )
        for slot, statistic in {
            "coupling" : "polarization",
            "heading"  : "turn_reach"
        }.items()
    }


def tabled(summary: Dataset) -> dict[str, dict[str, dict[str, float]]]:
    """
    Lays `summary` out by candidate and then by condition, each entry
    holding every variable `summary` carries.
    """
    return {
        str(candidate): {
            str(condition): {
                name: float(variable.sel(candidate=candidate, condition=condition))
                for name, variable in summary.data_vars.items()
            }
            for condition in summary.condition.values
        }
        for candidate in summary.candidate.values
    }
