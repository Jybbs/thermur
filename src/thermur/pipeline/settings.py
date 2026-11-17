"""
Holds `Settings`, every setting a run reads, which the `[tool.thermur]`
table of the clone's `pyproject.toml` sets for every run beneath a
`THERMUR_` variable and a value passed in.
"""

from jax               import vmap
from jax.numpy         import linspace
from pathlib           import Path
from pydantic          import model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from pydantic_settings.sources import PyprojectTomlConfigSettingsSource
from typing import Self

from thermur.controller.components import (
    Centered,
    Consensus,
    Constant,
    Cruise,
    HardSphere,
    Home,
    Zig
)
from thermur.controller.schemas  import Radio
from thermur.environment.plant   import Plant
from thermur.environment.schemas import Crop, ParametricThreat
from thermur.metrics.schemas     import Plan
from thermur.repo.checkout       import Checkout
from thermur.repo.schemas        import Manifest


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

    cohesion: Centered = Centered()
    """
    The acceleration holding a drone at a spacing from its neighbors.
    """

    contact: HardSphere = HardSphere()
    """
    The acceleration pushing a drone away from each of its neighbors.
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

    heading: Consensus = Consensus()
    """
    The acceleration turning a drone toward its neighbors.
    """

    noise: Zig = Zig()
    """
    The random increments on each drone's command and the zig each drone
    starts at the rim or takes up from its neighbors.
    """

    plan: Plan = Plan()
    """
    The flock's size, its launch, the flight's length and step, and the seeds
    it flies.
    """

    plant: Plant = Plant()
    """
    The lag and the bounds carrying a drone's velocity toward its command.
    """

    radio: Radio = Radio()
    """
    The link carrying each drone's state to the others.
    """

    roost: Home = Home()
    """
    The acceleration holding a drone over the roost.
    """

    simulation: Path | None = None
    """
    The WRF-SFIRE output the graph reads, relative to the clone, such as the
    `data/samples/wrf_sample.nc` the `thermur fetch sample` command writes,
    or `None` to read the dataset `threat` builds.
    """

    speed: Cruise = Cruise()
    """
    The acceleration relaxing a drone's speed toward the cruise speed.
    """

    threat: ParametricThreat = ParametricThreat()
    """
    The front of hot air the graph reads where `simulation` names no output.
    """

    @model_validator(mode="after")
    def confined(self) -> Self:
        """
        Holds one step of the integrator within the stability bound of
        the speed slot, as `plan.similarity` scales it, at every speed in
        `plan.airspeeds_m_s`, the step times the slot's stiffness staying at
        or below two.

        Raises:
            ValueError: Where `plan.step_s` times the stiffness at a speed
                        in the range exceeds two, past which one step widens
                        the gap between two nearly equal speeds there rather
                        than narrowing it.
        """
        if self.speed.fixed:
            return self

        plan      = self.plan
        stiffness = abs(
            vmap(self.speed.scaled(plan.similarity).stiffness, in_axes=(None, 0))(
                plan.cruise_m_s,
                linspace(*plan.airspeeds_m_s, 64)
            )
        ).max()
        if plan.step_s * stiffness > 2:
            raise ValueError(
                f"A step of {plan.step_s:g} s times the speed slot's stiffness "
                f"of {stiffness:.3g} per second exceeds two within "
                f"{plan.airspeeds_m_s[0]:g} to {plan.airspeeds_m_s[1]:g} m/s"
            )

        return self

    @model_validator(mode="after")
    def cruising(self) -> Self:
        """
        Holds the speed slot's cruise speed to `plan.cruise_m_s`, the speed
        the plan launches the flock at and a fixed flight holds each drone
        at.

        Raises:
            ValueError: Where the two cruise speeds differ.
        """
        if self.speed.cruise_m_s != self.plan.cruise_m_s:
            raise ValueError(
                f"The speed slot cruises at {self.speed.cruise_m_s:g} m/s, where "
                f"the plan launches the flock at {self.plan.cruise_m_s:g} m/s"
            )

        return self

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
