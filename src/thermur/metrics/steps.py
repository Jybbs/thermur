"""
Declares the flight's Hamilton steps, meaning the reference flight the
settings describe, its frames and its statistics as saved tables, one step
per open slot flying the slot's candidates against that reference, and the
sweep of flock sizes through still air.
"""

from hamilton.function_modifiers import parameterize, value
from xarray import DataTree, Dataset

from thermur.controller.schemas import Composite
from thermur.environment.air    import Air
from thermur.metrics.flight     import Flight
from thermur.metrics.sweep      import Sweep
from thermur.metrics.trials     import Trial
from thermur.pipeline.settings  import Settings


def flight(
    air            : Air,
    composite      : Composite,
    metrics_digest : str,
    settings       : Settings
) -> Flight:
    """
    Flies the flock the settings describe through the air.
    """
    return Flight.flown(air, composite, settings.plan)


def frames(flight: Flight, metrics_digest: str) -> Dataset:
    """
    Lays the flight's kept frames out as a table.
    """
    return flight.dataset


def statistics(flight: Flight, metrics_digest: str) -> Dataset:
    """
    Lays the flight's statistics out as a table.
    """
    return flight.statistics


@parameterize(
    **{trial.name: {"trial": value(trial)} for trial in Trial.listed()}
)
def flown_trial(
    air            : Air,
    composite      : Composite,
    metrics_digest : str,
    settings       : Settings,
    trial          : Trial
) -> Dataset:
    """
    Flies one open slot's candidates against the reference.
    """
    return trial.flown(air, composite, settings.plan)


def sweep(composite: Composite, metrics_digest: str, settings: Settings) -> DataTree:
    """
    Flies the composite through still air at each size of the sweep.
    """
    return Sweep().flown(composite, settings.plan)
