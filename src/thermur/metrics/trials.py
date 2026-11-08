"""
Holds `Trial`, one open slot's candidates flown against its reference over
one set of seeds, keeping each candidate only where its measurements favor
it, `Measure`, one measurement of a trial, which the `*_trial` steps read,
and `summarized`, the mean across seeds and its standard error that both the
trial and the rules page's charts read.
"""

from dataclasses import replace
from math        import inf
from numpy       import sqrt
from pydantic    import BaseModel, InstanceOf
from typing      import Self
from xarray      import DataArray, Dataset, concat

from thermur.controller.components import (
    Anisotropic,
    Centroid,
    Consensus,
    Constant,
    Delayed,
    Elastic,
    Fitted,
    Intrinsic,
    Metric,
    Persistent,
    Vectorial,
    Weighted
)
from thermur.controller.schemas import Component, Composite
from thermur.environment.air    import Air
from thermur.metrics.flight     import Flight
from thermur.metrics.schemas    import Plan

POLARIZATION = 0.96
"""
The polarization measured on starling flocks, which a flight is favored
nearest to.
"""

REFERENCE = "reference"
"""
The name of the composite the settings hold, which every candidate is
measured against.
"""

SIZES = {"large": {"drones": 200}, "middle": {"drones": 100}, "small": {"drones": 30}}
"""
The flock sizes the cohesion and noise trials fly, as changes to the plan by
the condition's name.
"""


class Measure(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One statistic of a trial and the value that favors a candidate.
    """

    statistic: str
    """
    The name of the variable of `Flight.statistics`.
    """

    condition: str | None = None
    """
    The condition the statistic reads, or `None` for its spread across every
    condition, which `goal` then reads as a flatness.
    """

    goal: float = 0.0
    """
    The value the statistic is favored nearest to, or positive or negative
    infinity where a larger or a smaller value is favored.
    """

    def score(self, table: Dataset) -> DataArray:
        """
        Reads the measurement from a trial's `table` so a larger value is
        favored, by candidate and seed.
        """
        values = table[self.statistic]
        picked = (
            values.std("condition") if self.condition is None
            else values.sel(condition=self.condition, drop=True)
        )
        means = picked.mean(
            [dim for dim in picked.dims if dim not in ("candidate", "seed")]
        )
        if abs(self.goal) == inf:
            return means if self.goal > 0 else -means

        return -abs(means - self.goal)


class Trial(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One open slot of the controller, its candidates, the conditions each
    flies under as changes to the plan, and the measurements choosing among
    them.
    """

    candidates: dict[str, InstanceOf[Component]]
    """
    Each candidate for the slot, by name.
    """

    conditions: dict[str, dict[str, int | float]]
    """
    The changes each condition makes to the plan, by the condition's name.
    """

    measures: tuple[Measure, ...]
    """
    The measurements deciding whether a candidate is favored.
    """

    slot: str
    """
    The name of the slot.
    """

    controls: dict[str, dict[str, InstanceOf[Component]]] = {}
    """
    The slots each control changes, by the control's name, reported beside the
    candidates and never kept.
    """

    duration_s: float = 60.0
    """
    The length of each flight of the trial, in seconds.
    """

    @property
    def name(self) -> str:
        """
        Names the Hamilton step flying the trial.
        """
        return f"{self.slot}_trial"

    def composites(self, composite: Composite) -> dict[str, Composite]:
        """
        Lays out each composite the trial flies by name, the reference
        first, then each candidate in the slot, then each control.
        """
        return {
            REFERENCE: composite,
            **{
                name: replace(composite, **{self.slot: candidate})
                for name, candidate in self.candidates.items()
            },
            **{
                name: replace(composite, **slots)
                for name, slots in self.controls.items()
            }
        }

    def favored(self, table: Dataset) -> DataArray:
        """
        Keeps each candidate no measure leaves worse than the reference
        and at least one leaves better, each difference compared with the
        standard error of the two means across seeds.

        Args:
            table: The statistics of every flight, by candidate, condition,
                   and seed.

        Returns:
            Whether each candidate is favored, by candidate.
        """
        scores = summarized(
            values = concat(
                [measure.score(table) for measure in self.measures],
                "measure"
            )
        )
        error  = scores["error"]
        margin = sqrt(error**2 + error.sel(candidate=REFERENCE) ** 2)
        gain   = scores["mean"] - scores["mean"].sel(candidate=REFERENCE)
        return (
            (gain >= -margin).all("measure")
            & (gain > margin).any("measure")
            & table.candidate.isin(list(self.candidates))
        )

    def flown(self, composite: Composite, air: Air, plan: Plan) -> Dataset:  # prose: ignore[unsorted-positionals]
        """
        Flies every composite under every condition over one set of seeds,
        then replays one short flight of each through each composite's
        drift, which shows whether likelihood ranking and force inference
        tell the composites apart on flights whose true composite is known.

        Returns:
            The statistics by candidate, condition, and seed, beside `favored`
            and the `log_likelihood` and `coefficient` of each model on each
            candidate's flight.
        """
        composites = self.composites(composite)
        long       = plan.changed(duration_s=self.duration_s)
        short      = plan.changed(
            duration_s  = 10.0,
            frame_every = 1,
            seeds       = 1,
            settled_s   = 0.0
        )
        table = concat(
            [
                concat(
                    [
                        Flight.flown(item, air, long.changed(**changes)).statistics
                        for changes in self.conditions.values()
                    ],
                    DataArray(list(self.conditions), dims="condition")
                ).assign(
                    **(
                        Flight.flown(item, air, short)
                              .likelihoods(composites, air).data_vars
                    )
                )
                for item in composites.values()
            ],
            DataArray(list(composites), dims="candidate")
        )
        return table.assign(favored=self.favored(table))

    @classmethod
    def listed(cls) -> list[Self]:
        """
        Lists each open slot's trial, flying the coupling's metric rule with
        its onset below any excess, so the rule reads metric from launch,
        and each cohesion candidate at the reference flock's spacing.
        """
        return [
            cls(
                candidates = {
                    "centroid" : Centroid(),
                    "elastic"  : Elastic(),
                    "fitted"   : Fitted()
                },
                conditions = SIZES,
                controls   = {"uncoupled": {"coupling": Constant(coupling_per_s=0.0)}},
                measures   = (
                    Measure(statistic="spacing_m"),
                    Measure(condition="middle", goal=inf, statistic="border_density"),
                    Measure(condition="middle", goal=inf, statistic="peak_height"),
                    Measure(condition="small", goal=1.0, statistic="fragments")
                ),
                slot = "cohesion"
            ),
            cls(
                candidates = {
                    "anisotropic" : Anisotropic(),
                    "metric"      : Metric(onset_k=-1.0),
                    "weighted"    : Weighted()
                },
                conditions = {
                    "dense"  : {"spacing_m": 10.0},
                    "middle" : {"spacing_m": 15.0},
                    "sparse" : {"spacing_m": 25.0}
                },
                measures = (
                    Measure(statistic="polarization"),
                    Measure(statistic="alignment_by_rank")
                ),
                slot = "coupling"
            ),
            cls(
                candidates = {"consensus": Consensus(), "delayed": Delayed()},
                conditions = {
                    "large"       : {"frame_every": 5, "turn_deg": 90.0},
                    "small"       : {"frame_every": 5, "turn_deg": 20.0},
                    "spontaneous" : {}
                },
                measures = (
                    Measure(condition="small", goal=1.0, statistic="turn_reach"),
                    Measure(condition="large", goal=1.0, statistic="turn_reach"),
                    Measure(condition="small", goal=inf, statistic="turn_speed_m_s"),
                    Measure(condition="large", goal=inf, statistic="turn_speed_m_s"),
                    Measure(
                        condition = "spontaneous",
                        goal      = inf,
                        statistic = "spectral_peak_hz"
                    ),
                    Measure(
                        condition = "spontaneous",
                        goal      = POLARIZATION,
                        statistic = "polarization"
                    )
                ),
                slot = "heading"
            ),
            cls(
                candidates = {
                    "intrinsic"  : Intrinsic(),
                    "persistent" : Persistent(),
                    "vectorial"  : Vectorial()
                },
                conditions = SIZES,
                measures   = (
                    *(
                        Measure(
                            condition = size,
                            goal      = POLARIZATION,
                            statistic = "polarization"
                        )
                        for size in SIZES
                    ),
                    Measure(condition="middle", goal=inf, statistic="turn_rate_per_s")
                ),
                slot = "noise"
            )
        ]


def summarized(values: DataArray) -> Dataset:
    """
    Reads the mean of `values` across seeds beside the standard error of
    that mean.

    Returns:
        The `mean` and the `error`, over every dimension of `values` but
        `seed`.
    """
    return Dataset(
        {
            "error" : values.std("seed") / sqrt(values.count("seed")),
            "mean"  : values.mean("seed")
        }
    )
