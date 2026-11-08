"""
Holds the stand-ins the controller's and the flight's cases share, meaning
one drone's view, a flock laid out by hand, a still parcel of air, and a
short plan.
"""

from jax.numpy import array, asarray, full, ones, zeros
from xarray    import Dataset

from thermur.controller.schemas import Flock, Observation
from thermur.environment.air    import Air
from thermur.metrics.schemas    import Plan


def flock_of(positions: object, velocities: object) -> Flock:
    """
    Builds a flock at `positions` and `velocities`, holding no force.
    """
    return Flock(
        forces     = zeros(asarray(positions).shape),
        positions  = asarray(positions),
        velocities = asarray(velocities)
    )


def short(**changes: object) -> Plan:
    """
    Builds the plan of a short flight of a small flock, carrying `changes`.
    """
    return Plan(
        **{
            "drones"           : 12,
            "duration_s"       : 2.0,
            "frame_every"      : 5,
            "seeds"            : 2,
            "settled_s"        : 0.5,
            "turn_at_s"        : 1.0,
            "turning_window_s" : 0.5,
            **changes
        }
    )


def still_air(excess: float = 0.0) -> Air:
    """
    Builds air holding one `excess` at every point of a small grid.
    """
    return Air.from_dataset(
        Dataset(
            {
                "excess": (
                    ("Time", "height", "south_north", "west_east"),
                    full((2, 2, 2, 2), excess)
                )
            },
            coords = {
                "Time"        : [0.0, 100.0],
                "height"      : [0.0, 1000.0],
                "south_north" : [0.0, 4000.0],
                "west_east"   : [0.0, 4000.0]
            }
        )
    )


def viewed(  # prose: ignore[unsorted-positionals]
    offsets  : list[list[float]],
    velocity : tuple[float, float, float] = (10.0, 0.0, 0.0),
    excess   : float = 0.0,
    **fields : object
) -> Observation:
    """
    Builds one drone's view of neighbors at `offsets`, each flying as
    the drone does and aligning at a rate of one, unless `fields` says
    otherwise.
    """
    spots = array(offsets)
    own   = array(velocity)
    return Observation(
        **{
            "air"       : {"excess": asarray(excess)},
            "delayed"   : own * ones((len(offsets), 3)),
            "distances" : (spots**2).sum(-1) ** 0.5,
            "force"     : zeros(3),
            "neighbors" : own * ones((len(offsets), 3)),
            "offsets"   : spots,
            "velocity"  : own,
            "weights"   : ones(len(offsets)),
            **fields
        }
    )
