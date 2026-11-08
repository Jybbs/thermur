"""
Holds `Settings`, every setting a run reads, which the `[tool.thermur]`
table of the clone's `pyproject.toml` sets for every run beneath a
`THERMUR_` variable and a value passed in.
"""

from cyclopts          import Parameter
from pathlib           import Path
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from pydantic_settings.sources import PyprojectTomlConfigSettingsSource
from typing import Annotated

from thermur.controller.components import Constant, Morse, Overdamped, SoftCore, White
from thermur.environment.schemas   import Crop, ParametricThreat
from thermur.metrics.schemas       import Plan
from thermur.repo.checkout         import Checkout
from thermur.repo.schemas          import Manifest


class Settings(
    BaseSettings,
    env_prefix                  = "THERMUR_",
    extra                       = "forbid",
    frozen                      = True,
    pyproject_toml_table_header = ("tool", "thermur"),
    use_attribute_docstrings    = True
):
    """
    Every setting a run reads, each a field carrying its default, its
    bound, and its unit, with each component of the controller nesting as an
    `equinox.Module` field whose own fields are its parameters.

    A value passed in takes precedence over the `THERMUR_` variable naming
    the field, which takes precedence over the `[tool.thermur]` table, and
    the model is frozen, so a node reading the settings cannot move a value
    another node has already read.
    """

    cohesion: Morse = Morse()
    """
    The acceleration holding a drone at a spacing from its neighbors.
    """

    contact: SoftCore = SoftCore()
    """
    The acceleration pushing apart two drones that overlap.
    """

    coupling: Constant = Constant()
    """
    The rate at which a drone aligns with each neighbor.
    """

    crop: Crop = Crop()
    """
    The window, the neighborhood, and the band a WRF-SFIRE output is cropped
    to, and the fixed heights its fields are regridded onto.
    """

    # `Overdamped` declares no field, which cyclopts renders as one flag taking a
    # value no conversion accepts, so the command line leaves the slot out.
    heading: Annotated[Overdamped, Parameter(parse=False)] = Overdamped()
    """
    The acceleration turning a drone toward its neighbors.
    """

    noise: White = White()
    """
    The random increments driving each drone's velocity and force.
    """

    plan: Plan = Plan()
    """
    The flock's size, its launch, the flight's length and step, and the seeds
    it flies.
    """

    simulation: Path | None = None
    """
    The WRF-SFIRE output the graph reads, relative to the clone, such as the
    `data/samples/wrf_sample.nc` the `thermur fetch sample` command writes,
    or `None` to read the dataset `threat` builds.
    """

    threat: ParametricThreat = ParametricThreat()
    """
    The front of hot air the graph reads where `simulation` names no output.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls  : type[BaseSettings],
        *,
        env_settings  : PydanticBaseSettingsSource,
        init_settings : PydanticBaseSettingsSource,
        **_           : PydanticBaseSettingsSource
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Orders the places a setting is read from, a value passed in first,
        a `THERMUR_` variable second, and the `[tool.thermur]` table third,
        leaving out the `.env` and secrets sources pydantic-settings orders
        by default.

        The table comes from the `pyproject.toml` of the clone
        `Checkout.installed` resolves, whatever the working directory.
        """
        return (
            init_settings, env_settings,
            PyprojectTomlConfigSettingsSource(
                settings_cls,
                Checkout.installed().root / Manifest.model_config["title"]
            )
        )
