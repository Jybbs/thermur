"""
Declares the flight's Hamilton steps, meaning the reference flight the
settings describe, its frames and its statistics as saved tables, and one
step per open slot flying the slot's candidates against that reference.
"""

from hamilton.function_modifiers import parameterize, value
from xarray import Dataset

from thermur.controller.schemas import Composite
from thermur.environment.air    import Air
from thermur.metrics.flight     import Flight
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
    return Flight.flown(composite, air, settings.plan)


def frames(flight: Flight) -> Dataset:
    """
    Lays the flight's kept frames out as a table.
    """
    return flight.dataset


def statistics(flight: Flight) -> Dataset:
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
    return trial.flown(composite, air, settings.plan)
