"""
Holds `Plan`, the flock's size, its launch, and the length, the step, and
the seeds of one flight, which the settings nest as `plan`.
"""

from jax        import Array
from jax.numpy  import array, cbrt, cos, deg2rad, pi, sin, stack, zeros_like
from jax.random import normal, split
from math       import radians
from numpy      import arange, ndarray
from pydantic   import BaseModel, Field, model_validator
from typing     import Annotated, Self

from thermur.controller.schemas  import Flock, unit
from thermur.environment.schemas import Count, NonNegative, Positive


class Plan(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One flight, flown from one launch per seed, every draw deriving from
    `seed`.
    """

    axes: tuple[Positive, Positive, Positive] = (5.6, 2.8, 1.0)
    """
    The launch cloud's length, width, and depth, in proportion, the thin
    horizontal shape measured on starling flocks, its long axis along the
    course and its thin axis vertical.
    """

    center_m: tuple[float, float, float] = (1000.0, 1000.0, 100.0)
    """
    The center of the launch cloud, east, north, and above the ground, in
    meters, by default on the parametric threat's course to the front it
    crosses within two minutes.
    """

    course_deg: float = 180.0
    """
    The launch heading in degrees counterclockwise from east, by default west.
    """

    cruise_m_s: Positive = 10.0
    """
    The speed a flight holds each drone at, in meters per second.
    """

    drones: Count = 100
    """
    The size of the flock.
    """

    duration_s: Positive = 120.0
    """
    The length of the flight, in seconds.
    """

    frame_every: Count = 25
    """
    The number of steps between two kept frames.
    """

    neighbors: Count = 7
    """
    The number of topological neighbors each drone senses.
    """

    resolution_s: Positive = 0.005
    """
    The time resolution of the Brownian path, in seconds, which a flight at
    a smaller `step_s` replays unchanged.
    """

    scatter_deg: Annotated[float, Field(ge=0, lt=90)] = 10.0
    """
    The spread of the launch headings about the course, in degrees.
    """

    seed: int = 0
    """
    The value every draw of a flight derives from.
    """

    seeds: Count = 4
    """
    The number of launches flown together under `vmap`.
    """

    settled_s: NonNegative = 20.0
    """
    The start of the flight a statistic leaves out, in seconds.
    """

    spacing_m: Positive = 15.0
    """
    The launch cloud's cube root of the volume per drone, in meters.
    """

    step_s: Positive = 0.02
    """
    The step of the integrator, in seconds.
    """

    turn_at_s: NonNegative = 25.0
    """
    The moment the first drone starts the turn, in seconds.
    """

    turn_deg: float = 0.0
    """
    The angle the first drone turns through and then holds, in degrees
    counterclockwise, none by default.
    """

    turning_deg_s: Positive = 2.0
    """
    The rotation of the flock's mean heading, over `turning_window_s`, past
    which a spontaneous turn counts as started, in degrees per second.
    """

    turning_window_s: Positive = 5.0
    """
    The time over which the flock's mean heading or a drone's heading is read
    for a turn, in seconds, longer than a step's noise on either.
    """

    @property
    def frame_s(self) -> float:
        """
        Multiplies the integrator's step by the steps between two frames.
        """
        return self.frame_every * self.step_s

    @property
    def frames(self) -> int:
        """
        Counts the frames a flight keeps.
        """
        return self.steps // self.frame_every

    @property
    def steps(self) -> int:
        """
        Counts the steps of the integrator.
        """
        return round(self.duration_s / self.step_s)

    @property
    def times(self) -> ndarray:
        """
        Lays out the time of each kept frame, at the end of its steps, in
        seconds.
        """
        return self.frame_s * arange(1, self.frames + 1)

    @property
    def turn_rad(self) -> float:
        """
        Converts the first drone's turn into radians.
        """
        return radians(self.turn_deg)

    @property
    def turned(self) -> Array:
        """
        Computes the velocity the first drone holds through its turn, in
        meters per second.
        """
        return self.cruise_m_s * horizontal(self.course_deg + self.turn_deg)

    @property
    def turning_rad_s(self) -> float:
        """
        Converts the rotation rate a spontaneous turn starts past into
        radians per second.
        """
        return radians(self.turning_deg_s)

    @property
    def window(self) -> int:
        """
        Counts the kept frames in `turning_window_s`, at least one.
        """
        return max(self.framed(self.turning_window_s), 1)

    def changed(self, **changes: object) -> Self:
        """
        Builds the plan carrying `changes`, validated as a plan built from
        scratch is, where `model_copy` validates nothing.
        """
        return self.model_validate(self.model_dump() | changes)

    def framed(self, seconds: float) -> int:
        """
        Counts the kept frames in `seconds`, to the nearest frame.
        """
        return round(seconds / self.frame_s)

    def launched(self, key: Array) -> Flock:
        """
        Draws a flock in the thin horizontal cloud `axes` shapes, as
        Gaussian offsets whose ellipsoid at one standard deviation
        holds `spacing_m` cubed for each drone, flying the course within
        `scatter_deg`.
        """
        where_key, heading_key = split(key)
        lengths                = array(self.axes) / cbrt(array(self.axes).prod())
        scale  = self.spacing_m * cbrt(array(self.drones * 3 / (4 * pi)))
        shape  = (self.drones, 3)
        course = horizontal(self.course_deg)
        basis  = stack(
            [course, horizontal(self.course_deg + 90), array([0.0, 0.0, 1.0])]
        )
        offsets = normal(where_key, shape) * lengths * scale @ basis
        scatter = normal(heading_key, shape) * deg2rad(array(self.scatter_deg))
        return Flock(
            forces     = zeros_like(offsets),
            positions  = array(self.center_m) + offsets,
            velocities = self.cruise_m_s * unit(course + scatter)
        )

    @model_validator(mode="after")
    def outnumbered(self) -> Self:
        """
        Holds the flock above the neighbor count, since `KDTree.query` fills
        a short row with an index past the last drone.

        Raises:
            ValueError: Where the flock holds no more drones than a drone
                        has neighbors.
        """
        if self.drones <= self.neighbors:
            raise ValueError(
                f"A flock of {self.drones} leaves too few drones for "
                f"{self.neighbors} neighbors each"
            )

        return self

    @model_validator(mode="after")
    def spanned(self) -> Self:
        """
        Holds one turning window past the settling, and the first drone's
        turn where the plan sets one, inside the flight, so every statistic
        reads at least one frame, and that turn after the first kept frame,
        which the turn's response reads its headings against.

        Raises:
            ValueError: Where `settled_s` plus `turning_window_s`, or
                        `turn_at_s` under a nonzero `turn_deg`, reaches
                        past the last kept frame, or where that turn starts
                        before the first.
        """
        last_s  = self.frames * self.frame_s
        checked = {
            "settled_s + turning_window_s": self.settled_s + self.turning_window_s
        } | (
            {"turn_at_s": self.turn_at_s} if self.turn_deg else {}
        )
        for name, value in checked.items():
            if value >= last_s:
                raise ValueError(
                    f"`{name}` of {value:g} s reaches past the last frame at "
                    f"{last_s:g} s"
                )

        if self.turn_deg and self.turn_at_s < self.frame_s:
            raise ValueError(
                f"`turn_at_s` of {self.turn_at_s:g} s starts before the first "
                f"frame at {self.frame_s:g} s"
            )

        return self


def horizontal(degrees: float) -> Array:
    """
    Points a horizontal unit vector `degrees` counterclockwise from east.
    """
    angle = deg2rad(degrees)
    return array([cos(angle), sin(angle), 0.0])
