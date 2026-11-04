"""
Pins `Air`, meaning the grid it lays a dataset's channels out on, the
values it clips and the spacings it refuses, and the readings `sampled`
interpolates at each drone's position and time.
"""

from equinox       import filter_jit
from jax           import numpy
from numpy         import arange, finfo, inf
from numpy.testing import assert_allclose
from pytest        import fixture, mark, param, raises
from xarray        import DataArray, Dataset

from thermur.environment.air import Air


def linear(**coordinates: list[float]) -> DataArray:
    """
    Lays out a field equal to the time plus twice the height, three times
    the northing, and four times the easting at every point of the grid
    `coordinates` places along each of `Air.axes`.
    """
    grid = {axis: DataArray(
        coordinates[axis],
        coords = {axis: coordinates[axis]},
        dims   = axis
    ) for axis in Air.axes}
    return (
        grid["Time"]
        + 2 * grid["height"]
        + 3 * grid["south_north"]
        + 4 * grid["west_east"]
    ).transpose(*Air.axes)


@fixture
def air() -> Air:
    """
    Lays out the linear field as the channel `linear` and its negation as
    `negated`, on snapshots 10 s apart, heights 5 m apart, and rows and
    columns 40 m apart.
    """
    field = linear(
        Time        = [0.0, 10.0],
        height      = [5.0, 10.0, 15.0],
        south_north = [20.0, 60.0],
        west_east   = [20.0, 60.0, 100.0]
    )
    return Air.from_dataset(Dataset({"linear": field, "negated": -field}))


def test_air_clips_an_infinite_value_to_half_the_largest_finite_one():
    """
    Asserts that an infinite value becomes half the largest finite value in
    single precision, keeping its sign.
    """
    coordinates = {
        "Time"        : [0.0],
        "height"      : [5.0],
        "south_north" : [20.0],
        "west_east"   : [20.0, 60.0]
    }
    field = linear(**coordinates).copy(data=[[[[inf, -inf]]]])

    channels = Air.from_dataset(Dataset({"field": field})).channels

    bound = finfo("float32").max / 2
    assert channels.ravel().tolist() == [bound, -bound]


def test_air_lays_each_variable_out_as_a_channel_on_the_grid(air: Air):
    """
    Asserts that each data variable becomes one channel in the order the
    dataset holds them, on the grid whose first point and spacing along each
    axis the coordinates set.
    """
    assert air.names == ("linear", "negated")
    assert air.channels.shape == (2, 2, 3, 2, 3)
    assert air.channels.dtype == numpy.float32
    assert (
        air.origin, air.spacing
    ) == ((0.0, 5.0, 20.0, 20.0), (10.0, 5.0, 40.0, 40.0))


def test_air_refuses_unevenly_spaced_points():
    """
    Asserts that points spaced unevenly along an axis are refused, naming
    the axis, since the grid holds one spacing per axis.
    """
    field = linear(
        Time        = [0.0],
        height      = [5.0, 10.0, 20.0],
        south_north = [20.0],
        west_east   = [20.0]
    )

    with raises(ValueError, match=r"\['height'\] are unevenly spaced"):
        Air.from_dataset(Dataset({"field": field}))


def test_air_spreads_a_variable_along_an_axis_it_lacks():
    """
    Asserts that a variable lacking one of the axes takes the same value
    along it, and that an axis holding one point takes a spacing of one.
    """
    coordinates = {
        "Time"        : [0.0],
        "height"      : [5.0, 10.0],
        "south_north" : [20.0],
        "west_east"   : [20.0]
    }
    field  = linear(**coordinates)
    ground = field.isel(drop=True, height=0)

    air = Air.from_dataset(Dataset({"field": field, "ground": ground}))

    assert air.spacing == (1.0, 5.0, 1.0, 1.0)
    assert_allclose(air.channels[1, 0, :, 0, 0], [ground.item()] * 2)


def test_sampling_holds_every_reading_over_clipped_values_finite():
    """
    Asserts that each reading over a grid holding only clipped values, or
    clipped values beside finite ones, stays finite, along a diagonal whose
    weights round a sum of the largest finite value past it.
    """
    field = linear(
        Time        = [0.0, 15.0],
        height      = [20.0, 40.0],
        south_north = [20.0, 60.0],
        west_east   = [20.0, 60.0]
    )
    air = Air.from_dataset(
        Dataset(
            {"beside": field.where(field.west_east > 20.0, inf), "only": field * inf}
        )
    )
    along = numpy.linspace(20.0, 60.0, 41)

    readings = air.sampled(numpy.stack([along, along, along / 2 + 10], axis=1), 7.3)

    assert all(numpy.isfinite(reading).all() for reading in readings.values())


def test_sampling_holds_the_edge_value_beyond_the_grid(air: Air):
    """
    Asserts that a drone outside the grid, or a time past the last snapshot,
    reads the value at the nearest edge of the grid.
    """
    readings = air.sampled(numpy.array([[500.0, -100.0, 50.0]]), 60.0)

    assert_allclose(readings["linear"], [10.0 + 2 * 15.0 + 3 * 20.0 + 4 * 100.0])


def test_sampling_interpolates_linearly_between_the_points(air: Air):
    """
    Asserts that each channel read at a drone's position between the grid's
    points takes the linear field's value there, under `jit`, in time as
    well as in space.
    """
    positions = numpy.array([[40.0, 30.0, 7.5], [90.0, 50.0, 12.0]])

    readings = filter_jit(air.sampled)(positions, 4.0)

    expected = [
        4.0 + 2 * 7.5 + 3 * 30.0 + 4 * 40.0,
        4.0 + 2 * 12.0 + 3 * 50.0 + 4 * 90.0
    ]
    assert_allclose(readings["linear"], expected, rtol=1e-6)
    assert_allclose(readings["negated"], [-value for value in expected], rtol=1e-6)


def test_sampling_reads_an_axis_of_one_point_as_that_point_everywhere():
    """
    Asserts that a drone reads an axis holding one point as that point at
    any coordinate along it, here the time and the height, while it still
    interpolates along an axis holding more.
    """
    field = linear(
        Time        = [30.0],
        height      = [5.0],
        south_north = [20.0],
        west_east   = [20.0, 60.0]
    )

    readings = Air.from_dataset(Dataset({"field": field})).sampled(
        numpy.array([[40.0, 900.0, 77.0]]),
        500.0
    )

    assert_allclose(readings["field"], [30.0 + 2 * 5.0 + 3 * 20.0 + 4 * 40.0])


@mark.parametrize(
    "drones",
    [param(0, id="no-drones"), param(1, id="one-drone"), param(4, id="four-drones")]
)
def test_sampling_reads_one_value_per_drone(air: Air, drones: int):
    """
    Asserts that each channel's reading holds one value per drone, no value
    where the flock holds no drone.
    """
    steps     = arange(drones, dtype=float)
    positions = numpy.stack([steps * 20, steps * 10, steps + 5], axis=1)

    readings = air.sampled(positions, 0.0)

    assert {name: reading.shape for name, reading in readings.items()} == {
        "linear"  : (drones,),
        "negated" : (drones,)
    }
