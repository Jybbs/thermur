"""
Holds `Sweep`, the flock flown through still air at each of several sizes
over one set of seeds, beside the same flock held at its cruise speed, which
the `sweep` step reads, and the slope of each statistic against the size on
logarithmic axes.
"""

from dataclasses import replace
from numpy       import asarray, isfinite, log
from pydantic    import BaseModel
from scipy.stats import linregress
from xarray      import DataTree, Dataset

from thermur.controller.schemas  import Composite
from thermur.environment.air     import Air
from thermur.environment.schemas import Count, Positive
from thermur.metrics.flight      import Flight
from thermur.metrics.schemas     import Plan

CONTROL = "fixed"
"""
The name of the child under each size holding the flight whose speed slot
holds each drone at the cruise speed.
"""


class Sweep(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The sizes, the seeds, and the run length of a sweep through still air,
    each size's run long enough to span `spans` of the slower of the two
    relaxation times a short pilot flight of that size shows.
    """

    longest_s: Positive = 600.0
    """
    The longest run past the settling a size flies, in seconds, which holds
    the sweep's wall time and the frames a size keeps in memory bounded.
    """

    pilot_s: Positive = 120.0
    """
    The length of the pilot flight each size's relaxation times are read
    from, in seconds.
    """

    pilot_seeds: Count = 2
    """
    The launches each size's pilot flight flies.
    """

    seeds: Count = 10
    """
    The launches each size flies together under `vmap`.
    """

    sizes: tuple[Count, ...] = (10, 30, 100, 300, 1000, 3000)
    """
    The flock sizes, about evenly spaced on a logarithmic axis.
    """

    spans: Positive = 10.0
    """
    The relaxation times a run spans past the settling.
    """

    def flown(self, composite: Composite, plan: Plan) -> DataTree:
        """
        Flies each size through still air, first over a pilot flight whose
        heading autocorrelation and neighbor rearrangement times set the
        run's length, then over every seed with `composite`, and then with
        its speed slot holding each drone at the cruise speed.

        Returns:
            The still-air statistics of each size's flight under the size,
            beside the control's under `CONTROL`, each flight's length and
            the pilot's two relaxation times kept as the size's attributes.
        """
        air   = Air.still()
        fixed = replace(composite, speed=replace(composite.speed, fixed=True))
        nodes = {}
        for size in self.sizes:
            sized = plan.changed(drones=size, seeds=self.seeds)
            pilot = Flight.flown(
                air,
                composite,
                sized.changed(duration_s=self.pilot_s, seeds=self.pilot_seeds)
            )
            times = {
                "autocorrelation_s" : float(pilot.autocorrelation_s.mean()),
                "rearrangement_s"   : float(pilot.rearrangement_s.mean())
            }
            run = sized.changed(
                duration_s = sized.settled_s
                + min(self.spans * max(times.values()), self.longest_s)
            )
            nodes[str(size)] = Flight.flown(
                air,
                composite,
                run
            ).still_statistics.assign_attrs(
                drones     = size,
                duration_s = run.duration_s,
                **times
            )
            nodes[f"{size}/{CONTROL}"] = Flight.flown(air, fixed, run).still_statistics

        return DataTree.from_dict(nodes)

    @staticmethod
    def slopes(tree: DataTree) -> Dataset:
        """
        Fits each statistic of the composite's flights against the size
        on logarithmic axes, across every seed of every size, through
        `linregress`, leaving out a seed whose statistic is `nan`.

        Returns:
            The slope and its standard error by statistic, for each
            statistic positive wherever it is measured and measured at two
            sizes or more.
        """
        tables = {
            int(node.attrs["drones"]): node.to_dataset()
            for node in tree.children.values()
        }
        points = {
            name: [
                (size, value)
                for size, table in tables.items()
                for value in table[name].values
                if isfinite(value)
            ]
            for name, variable in next(iter(tables.values())).data_vars.items()
            if variable.dims == ("seed",)
        }
        fits = {
            name: linregress(*log(asarray(pairs)).T)
            for name, pairs in points.items()
            if len({size for size, _ in pairs}) > 1 and (asarray(pairs) > 0).all()
        }
        return Dataset(
            {
                "error" : ("statistic", [fit.stderr for fit in fits.values()]),
                "slope" : ("statistic", [fit.slope for fit in fits.values()])
            },
            coords = {"statistic": list(fits)}
        )
