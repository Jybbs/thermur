"""
Holds `measured`, which reads the means and standard errors the charts on
The Interaction Rules draw out of the coupling and heading trials a run
saved, `still_air`, which reads those The Flock in Still Air draws out
of the sweep a run saved, and `MEASURED` and `STILL_AIR`, the files the
`docs:measure` task writes them into.
"""

from math        import isnan
from pathlib     import Path
from scipy.stats import spearmanr
from xarray      import Dataset, open_dataset, open_datatree

from thermur.metrics.norms  import Norms
from thermur.metrics.sweep  import CONTROL, Sweep
from thermur.metrics.trials import Trial, summarized

MEASURED = Path("docs/.vitepress/lib/measured/trials.json")
"""
The file the charts read the trials' means from, relative to the clone's
root.
"""

STILL_AIR = MEASURED.with_name("still-air.json")
"""
The file the still-air page's figures read the sweep's means from,
relative to the clone's root.
"""

DIGITS = 4
"""
The decimal places each mean and standard error keeps.
"""


def kept(value: float) -> float | None:
    """
    Leaves `value` as it is, or `None` where it is `nan`, which JSON holds
    no literal for.
    """
    return None if isnan(value) else value


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


def still_air(run: Path) -> dict[str, object]:
    """
    Reads each statistic's mean and standard error across seeds at each
    size of the sweep a run saved, for the composite and for its fixed-speed
    control, beside each size's run length and pilot relaxation times,
    each norm `Norms` declares, each statistic's slope against the size on
    logarithmic axes, and the rank correlation of the group speed with the
    size across every flight.

    Args:
        run: The run's folder under `data/runs/`, holding the `sweep.nc`
             that `thermur fly sweep` saves.
    """
    tree   = open_datatree(run / "sweep.nc")
    sizes  = sorted(int(node.attrs["drones"]) for node in tree.children.values())
    tables = {
        flight: {
            size: (
                tree[str(size) if flight == "flocking" else f"{size}/{CONTROL}"]
                .to_dataset()
                .astype("float64")
            )
            for size in sizes
        }
        for flight in ("flocking", CONTROL)
    }
    flocking = tables["flocking"]
    speeds   = [
        (size, float(value))
        for size, table in flocking.items()
        for value in table["group_speed_m_s"].values
    ]
    slopes = Sweep.slopes(tree).round(DIGITS)
    ranked = spearmanr(*zip(*speeds, strict=True))
    return {
        "norms"         : Norms.shipped().model_dump(mode="json"),
        "runs"          : {
            str(size): {
                name: round(float(tree[str(size)].attrs[name]), DIGITS)
                for name in ("autocorrelation_s", "duration_s", "rearrangement_s")
            }
            for size in sizes
        },
        "sizes"         : sizes,
        "slopes"        : {
            str(name): {
                "error" : float(slopes.error.sel(statistic=name)),
                "slope" : float(slopes.slope.sel(statistic=name))
            }
            for name in slopes.statistic.values
        },
        "speed_by_size" : {
            "p"   : round(float(ranked.pvalue), DIGITS),
            "rho" : round(float(ranked.statistic), DIGITS)
        },
        "statistics"    : {
            str(name): {
                flight: [
                    {"size": size} | {
                        key: kept(float(value))
                        for key, value in (
                            summarized(table[size][name])
                            .round(DIGITS)
                            .data_vars.items()
                        )
                    }
                    for size in sizes
                ]
                for flight, table in tables.items()
            }
            for name, variable in flocking[sizes[0]].data_vars.items()
            if variable.dims == ("seed",)
        }
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
