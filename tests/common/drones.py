"""
Holds the stand-ins the controller's, the environment's, the flight's, and
the graph's cases share, meaning one drone's view, a flock laid out by hand,
a parcel of air holding one wind, a short plan, and the settings of a small
flight.
"""

from jax.numpy import array, asarray, full, linalg, ones, zeros
from xarray    import Dataset

from thermur.controller.schemas  import Flock, Observation, unit
from thermur.environment.air     import Air
from thermur.environment.schemas import Crop, Grid, ParametricThreat
from thermur.metrics.schemas     import Plan
from thermur.pipeline.settings   import Settings


def flock_of(positions: object, velocities: object, commands: object = None) -> Flock:
    """
    Builds a flock at `positions` and `velocities`, holding no force and
    no spin, each drone commanding `commands`, or its own velocity where
    `commands` is `None`.
    """
    return Flock(
        commands   = asarray(velocities if commands is None else commands),
        forces     = zeros(asarray(positions).shape),
        positions  = asarray(positions),
        spins      = zeros(asarray(positions).shape),
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


def small() -> Settings:
    """
    Builds the settings of a short flight through the parametric threat
    on a small grid, which the crop's fixed heights from 20 m to 100 m sit
    inside.
    """
    return Settings(
        crop   = Crop(floor_m=20.0, levels=5, spacing_m=20.0),
        plan   = short(center_m=(160.0, 40.0, 60.0)),
        threat = ParametricThreat(
            edge_m = 10.0,
            grid   = Grid(
                columns   = 8,
                levels    = 10,
                rows      = 2,
                snapshots = 2
            )
        )
    )


def still_air(
    excess : float = 0.0,
    wind   : tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Air:
    """
    Builds air holding one `excess` and one `wind`, east, north, and up, at
    every point of a small grid.
    """
    return Air.from_dataset(
        Dataset(
            {
                name: (
                    ("Time", "height", "south_north", "west_east"),
                    full((2, 2, 2, 2), value)
                )
                for name, value in zip(
                    ("excess", "u", "v", "w"),
                    (excess, *wind),
                    strict = True
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


def viewed(
    offsets  : list[list[float]],
    excess   : float = 0.0,
    velocity : tuple[float, float, float] = (10.0, 0.0, 0.0),
    **fields : object
) -> Observation:
    """
    Builds one drone's view of neighbors at `offsets`, each flying as the
    drone does, resting, and aligning at a rate of one, the drone at the
    center of the default launch and its radio holding those neighbors
    alone, unless `fields` says otherwise.
    """
    spots     = array(offsets)
    own       = array(velocity)
    distances = linalg.norm(spots, axis=-1)
    return Observation(
        **{
            "air": {"excess": asarray(excess)} | dict.fromkeys("uvw", asarray(0.0)),
            "centrality" : linalg.norm(unit(spots).sum(0))
            / max(int((distances > 0).sum()), 1),
            "cruise"     : linalg.norm(own),
            "delayed"    : own * ones((len(offsets), 3)),
            "distances"  : distances,
            "force"      : zeros(3),
            "forces"     : zeros((len(offsets), 3)),
            "neighbors"  : own * ones((len(offsets), 3)),
            "offsets"    : spots,
            "position"   : array(Plan().center_m),
            "spin"       : zeros(3),
            "velocity"   : own,
            "weights"    : ones(len(offsets)),
            **fields
        }
    )
