"""
Holds `Air`, the channels of a WRF-SFIRE output laid out on a uniform
`(time, height, y, x)` grid, which a flight samples at each drone's position
inside `jit` and `vmap`. `WrfOutput.air` builds it, and the `air` step reads
it.
"""

from equinox           import Module, field
from functools         import partial
from jax               import Array, vmap
from jax.numpy         import asarray, full_like
from jax.scipy.ndimage import map_coordinates
from jax.typing        import ArrayLike
from numpy             import allclose, diff, finfo
from typing            import ClassVar, Self
from xarray            import Dataset


class Air(Module):
    """
    The channels a flight reads, each on one uniform grid of snapshots,
    fixed heights above the ground, rows, and columns, whose first point
    and spacing along each axis are static, so `jit` compiles one flight
    per grid.
    """

    axes: ClassVar[tuple[str, ...]] = ("Time", "height", "south_north", "west_east")
    """
    The dimensions the grid lays its points out along, in the order of the
    axes of `channels` after the first.
    """

    channels: Array
    """
    The value of every channel at every point of the grid, `(n_channels,
    n_times, n_heights, n_rows, n_columns)`.
    """

    names: tuple[str, ...] = field(static=True)
    """
    The name of each channel, in the order `channels` holds them.
    """

    origin: tuple[float, float, float, float] = field(static=True)
    """
    The grid's first point along each axis, in seconds since the simulation
    started and in meters above the ground, north of the southern edge, and
    east of the western edge.
    """

    spacing: tuple[float, float, float, float] = field(static=True)
    """
    The distance between neighboring points along each axis, in the unit of
    `origin` along it.
    """

    @classmethod
    def from_dataset(cls, channels: Dataset) -> Self:
        """
        Lays out each data variable of `channels` as one channel, on the
        dimensions `axes` names, a variable lacking one of them taking the
        same value along it.

        An infinite value becomes half the largest finite single-precision
        value, keeping its sign, so a reading stays finite, since the
        weighted sum `map_coordinates` takes of values at the largest
        one rounds past it to infinity. An axis holding one point takes
        a spacing of one, which `mode="nearest"` reads as that point
        everywhere along the axis.

        Raises:
            ValueError: Where the points along an axis are unevenly spaced.
        """
        stacked = channels.to_dataarray("channel").transpose("channel", *cls.axes)
        steps   = {axis: diff(stacked[axis].values) for axis in cls.axes}
        uneven  = [axis for axis, step in steps.items() if not allclose(step, step[:1])]
        if uneven:
            raise ValueError(f"The points along {uneven} are unevenly spaced")

        bound = finfo("float32").max / 2
        return cls(
            channels = asarray(stacked.values.clip(-bound, bound), dtype="float32"),
            names    = tuple(channels.data_vars),
            origin   = tuple(float(stacked[axis][0]) for axis in cls.axes),
            spacing  = tuple(
                float(step[0]) if step.size else 1.0 for step in steps.values()
            )
        )

    def sampled(self, positions: Array, t: ArrayLike) -> dict[str, Array]:
        """
        Reads every channel at each drone's position at the time `t`,
        through one `map_coordinates` call per channel under `vmap`, which
        interpolates linearly between the grid's points and holds the value
        at the grid's edge beyond it.

        Args:
            positions : Each drone's east, north, and height above the
                        ground, in meters, `(n_drones, 3)`.
            t         : The time, in seconds since the simulation started.

        Returns:
            Each channel's reading at each drone, `(n_drones,)`, by the
            channel's name.
        """
        x, y, z = positions.T
        indices = [
            (point - start) / step
            for point, start, step in zip(
                (full_like(x, t), z, y, x),
                self.origin,
                self.spacing,
                strict = True
            )
        ]
        readings = vmap(
            partial(map_coordinates, coordinates=indices, mode="nearest", order=1)
        )(
            self.channels
        )
        return dict(zip(self.names, readings, strict=True))
