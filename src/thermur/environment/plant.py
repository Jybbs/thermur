"""
Holds `Plant`, the autopilot carrying each drone's velocity toward the
velocity its controller commands, under the speed and acceleration bounds of
a multicopter, which the composite nests and a flight steps.
"""

from equinox   import Module
from jax       import Array
from jax.numpy import clip, concatenate, finfo, linalg, maximum, minimum
from typing    import Self

from thermur.environment.schemas import Positive, documented


@documented
class Plant(Module):
    """
    The first-order lag from a drone's command to its velocity, bounding
    the command by the speeds and the response by the accelerations the PX4
    multicopter defaults set, with the horizontal speed raised within PX4's
    range to the fastest starling flock measured, the descent's speed and
    acceleration raised within PX4's ranges to match the climb's, and no
    gravity term, since the autopilot carries the drone's weight.

    The vertical speed bounds are even because a clip wider above than below
    turns noise of zero mean on a vertical command into a mean climb.
    """

    climb_m_s: Positive = 3.0
    """
    The fastest climb a command holds, in meters per second, PX4's
    `MPC_Z_VEL_MAX_UP`.
    """

    climbing_m_s2: Positive = 4.0
    """
    The largest upward acceleration, in meters per second squared, PX4's
    `MPC_ACC_UP_MAX`.
    """

    descending_m_s2: Positive = 4.0
    """
    The largest downward acceleration, in meters per second squared, PX4's
    `MPC_ACC_DOWN_MAX`, raised from its default of 3 within the 2 to 15 PX4
    allows to match `climbing_m_s2`.
    """

    descent_m_s: Positive = 3.0
    """
    The fastest descent a command holds, in meters per second, PX4's
    `MPC_Z_VEL_MAX_DN`, raised from its default of 1.5 within the 0.5 to 4
    PX4 allows to match `climb_m_s`.
    """

    lag_s: Positive = 1.0
    """
    The time constant over which a drone's velocity relaxes toward its
    command, in seconds.
    """

    level_m_s: Positive = 15.2
    """
    The fastest horizontal speed a command holds, in meters per second,
    within the 0 to 20 PX4 allows its `MPC_XY_VEL_MAX`.
    """

    level_m_s2: Positive = 5.0
    """
    The largest horizontal acceleration, in meters per second squared,
    the ceiling PX4's `MPC_ACC_HOR_MAX` sets in its direct-velocity mode.
    """

    def held(self, commands: Array) -> Array:
        """
        Holds each command within the speed bounds, the horizontal bound
        on the length of the horizontal part and the vertical bounds on the
        height's rate.
        """
        return capped(
            high    = self.climb_m_s,
            level   = self.level_m_s,
            low     = -self.descent_m_s,
            vectors = commands
        )

    def scaled(self, _: float) -> Self:
        """
        Leaves the plant as it stands, since its bounds are a drone's own at
        any spacing.
        """
        return self

    def tracked(self, commands: Array, velocities: Array) -> Array:
        """
        Computes the acceleration carrying each velocity toward its command,
        the command first held within the speed bounds and the acceleration
        then held within the acceleration bounds, each horizontal bound on
        the length of the horizontal part and each vertical bound on the
        height's rate.

        Args:
            commands   : Each drone's commanded velocity, `(..., 3)`, in
                         meters per second.
            velocities : Each drone's velocity, `(..., 3)`, in meters per
                         second.

        Returns:
            Each drone's acceleration, `(..., 3)`, in meters per second
            squared.
        """
        return capped(
            high    = self.climbing_m_s2,
            level   = self.level_m_s2,
            low     = -self.descending_m_s2,
            vectors = (self.held(commands) - velocities) / self.lag_s
        )


def capped(high: float, level: float, low: float, vectors: Array) -> Array:
    """
    Shortens the horizontal part of each vector to at most `level`, keeping
    its direction, and clips the vertical part between `low` and `high`.
    """
    across = vectors[..., :2]
    length = linalg.norm(across, axis=-1, keepdims=True)
    return concatenate(
        [
            across * minimum(1.0, level / maximum(length, finfo(length.dtype).tiny)),
            clip(vectors[..., 2:], low, high)
        ],
        axis = -1
    )
