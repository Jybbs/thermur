"""
Pins the conversions `WrfOutput` takes of a WRF-SFIRE output, read here off
the dataset a parametric threat builds:

- The air temperature, the height above the ground, and the destaggered
  winds on the model's levels, and the range each is checked against
- The refusal of a missing or fill value in the ground heat flux and the
  smoke tracer, which hold no range
- The regridding onto fixed heights, the excess over the ambient profile,
  and the distance to the front
- The air they lay out, the crop each constructor applies, and the
  functions applied along one dimension or one snapshot
- The figures a plume measures, the sample cut over its window, and the
  file a cut writes
"""

from contextlib     import AbstractContextManager, nullcontext
from numpy          import arange, array, inf, isfinite, isnan, nan, sqrt, zeros
from numpy.testing  import assert_allclose
from pathlib        import Path
from pytest         import MonkeyPatch, fixture, mark, param, raises
from xarray         import DataArray, Dataset, full_like, open_dataset, zeros_like
from xarray.testing import assert_identical

from thermur.environment.schemas import (
    BASE_K,
    Crop,
    DRY_AIR_J_KG_K,
    GRAVITY_M_S2,
    Grid,
    HEAT_CAPACITY_J_KG_K,
    ParametricThreat,
    REFERENCE_PA,
    Sample,
    Span
)
from thermur.environment.wrf   import (
    WrfOutput,
    checked,
    destaggered,
    distances,
    extremes,
    spans,
    unfilled
)
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout

# The fill value netCDF writes by default in single precision, `NC_FILL_FLOAT`.
FILL_VALUE = 9.969209968386869e36


@fixture
def crop() -> Crop:
    """
    Builds the crop each case reads its output through, laying out the fixed
    heights from 20 m to 100 m.
    """
    return Crop(floor_m=20.0, levels=5, spacing_m=20.0)


@fixture
def threat() -> ParametricThreat:
    """
    Builds a threat on a grid of 20 columns, 3 rows, and 12 levels 20 m
    deep, over 3 snapshots 15 s apart, whose front advances 10 m each second
    behind an edge of scale 10 m.
    """
    return ParametricThreat(
        edge_m = 10.0,
        grid   = Grid(
            columns   = 20,
            levels    = 12,
            rows      = 3,
            snapshots = 3
        ),
        speed_m_s = 10.0
    )


@fixture
def developing(tmp_path: Path) -> Path:
    """
    Writes a threat whose front advances 2 m each second over 10 snapshots
    15 s apart to disk, standing in for a full simulation a sample is cut
    from.
    """
    path = tmp_path / Sample().simulation
    ParametricThreat(
        edge_m = 10.0,
        grid   = Grid(
            columns   = 20,
            levels    = 12,
            rows      = 3,
            snapshots = 10
        ),
        speed_m_s = 2.0
    ).dataset.to_netcdf(path, engine="netcdf4")
    return path


@fixture
def output(crop: Crop, threat: ParametricThreat) -> WrfOutput:
    """
    Wraps the dataset `threat` builds, cropped through `crop`.
    """
    return WrfOutput.from_threat(threat, crop=crop)


def exner(heights: DataArray) -> DataArray:
    """
    Measures the ratio of the threat's base-state air temperature to its
    potential temperature at each of `heights`, which falls by `g / c_p` per
    meter over `BASE_K`.
    """
    return 1 - GRAVITY_M_S2 * heights / (HEAT_CAPACITY_J_KG_K * BASE_K)


def scaled(threat: ParametricThreat, *, scale: float) -> Dataset:
    """
    Scales the base-state geopotential of one column of the threat's
    dataset, leaving every other column on its own levels.
    """
    data         = threat.dataset
    corner       = {"south_north": 0, "west_east": 0}
    base         = data.PHB.copy()
    base[corner] = base[corner] * scale
    return data.assign(PHB=base)


def test_a_crop_drops_every_other_coordinate(crop: Crop, threat: ParametricThreat):
    """
    Asserts that a crop keeps none of the coordinates an output holds beyond
    the ones it places, such as the `XLAT` a WRF-SFIRE output carries.
    """
    data = threat.dataset

    placed = WrfOutput.from_dataset(data.assign_coords(XLAT=data.GRNHFX), crop=crop)

    assert "XLAT" not in placed.dataset.coords


@mark.parametrize(
    ("crop", "empty"),
    [
        param(Crop(window_s=(100.0, 200.0)), r"\['Time'\]", id="a-window"),
        param(Crop(window_s=(30.0, 0.0)), r"\['Time'\]", id="a-reversed-window"),
        param(Crop(x_m=(5.0, 10.0)), r"\['west_east'\]", id="a-neighborhood"),
        param(
            Crop(x_m=(300.0, 100.0)),
            r"\['west_east', 'west_east_stag'\]",
            id = "a-reversed-neighborhood"
        )
    ]
)
def test_a_crop_keeping_no_point_along_a_dimension_is_refused(
    crop   : Crop,
    empty  : str,
    threat : ParametricThreat
):
    """
    Asserts that a window holding no snapshot, or a neighborhood holding no
    mass point, is refused, naming the dimension it empties, bounds given
    from high to low among them.
    """
    with raises(ValueError, match=f"The crop keeps no point along {empty}"):
        WrfOutput.from_threat(threat, crop=crop)


def test_a_crop_keeps_the_band_above_raised_ground(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that a crop over ground 1,000 m above sea level keeps the levels
    it keeps over ground at sea level, since the band is a height above
    the ground.
    """
    data   = threat.dataset
    raised = WrfOutput.from_dataset(
        data.assign(PHB=data.PHB + GRAVITY_M_S2 * 1000.0),
        crop = crop
    )

    assert raised.dataset.sizes["bottom_top"] == 7
    assert raised.aloft.sizes["height"] == 5


def test_a_crop_keeps_the_levels_its_shortest_column_needs(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that a crop reads its levels from the column reaching the top of
    the band last, so one column half as deep still covers the band.
    """
    output = WrfOutput.from_dataset(scaled(threat, scale=0.5), crop=crop)

    assert output.aloft.sizes["height"] == 5


def test_a_cut_from_another_output_than_the_sample_s_is_refused(
    crop     : Crop,
    tmp_path : Path
):
    """
    Asserts that a sample is cut only from the output it names, refusing
    another before opening it.
    """
    with raises(ValueError, match="`wrfout.nc` is not `wrfout_W5F7R4`"):
        WrfOutput.from_sample(
            tmp_path / "wrfout.nc",
            sample   = Sample(),
            settings = Settings(crop=crop)
        )


def test_a_cut_is_written_compressed_and_reads_back_as_the_output_cut(
    crop     : Crop,
    output   : WrfOutput,
    tmp_path : Path
):
    """
    Asserts that a written output reads back through `from_path` as the
    dataset it held, each data variable compressed with zlib, `XTIME`
    holding each snapshot's time in minutes, and WRF's grid attributes
    giving the sizes of the levels, the rows, and the columns the crop keeps
    rather than those of the output it was cut from.
    """
    path = output.write(tmp_path / "samples" / "wrf_sample.nc")

    with open_dataset(path, engine="netcdf4") as written:
        assert all(written[name].encoding["zlib"] for name in written.data_vars)
        assert written.XTIME.values.tolist() == [0.0, 0.25, 0.5]

    assert_identical(
        WrfOutput.from_path(path, crop=crop).dataset.load(),
        output.dataset.assign_attrs(
            {
                "BOTTOM-TOP_GRID_DIMENSION"    : 8,
                "BOTTOM-TOP_PATCH_END_STAG"    : 8,
                "BOTTOM-TOP_PATCH_END_UNSTAG"  : 7,
                "SOUTH-NORTH_GRID_DIMENSION"   : 4,
                "SOUTH-NORTH_PATCH_END_STAG"   : 4,
                "SOUTH-NORTH_PATCH_END_UNSTAG" : 3,
                "WEST-EAST_GRID_DIMENSION"     : 21,
                "WEST-EAST_PATCH_END_STAG"     : 21,
                "WEST-EAST_PATCH_END_UNSTAG"   : 20
            }
        )
    )


def test_a_cut_writes_its_snapshots_at_a_fixed_size(
    crop     : Crop,
    threat   : ParametricThreat,
    tmp_path : Path
):
    """
    Asserts that a cut of an output read from a file whose `Time` is
    unlimited writes `Time` at a fixed size, so the file depends only on the
    values it holds.
    """
    source = tmp_path / "wrfout.nc"
    threat.dataset.to_netcdf(source, engine="netcdf4", unlimited_dims=["Time"])
    path = WrfOutput.from_path(source, crop=crop).write(tmp_path / "wrf_sample.nc")

    with open_dataset(path, engine="netcdf4") as written:
        assert not written.encoding["unlimited_dims"]


def test_a_sample_is_cut_over_its_window_and_checked(crop: Crop, developing: Path):
    """
    Asserts that a sample is cut from the full output over the window its
    times bound, under the settings' crop, and taken where the window holds
    its figures over a two-minute flight.
    """
    sample = Sample(
        burning_m2 = (4800.0, 33600.0),
        hottest_k  = 319.8,
        times_s    = tuple(15.0 * snapshot for snapshot in range(1, 10))
    )

    cut = WrfOutput.from_sample(developing, sample=sample, settings=Settings(crop=crop))

    assert cut.dataset.Time.values.tolist() == list(sample.times_s)
    assert cut.dataset.sizes["west_east"] == 20


def test_a_sample_whose_window_misses_its_figures_is_refused(
    crop       : Crop,
    developing : Path
):
    """
    Asserts that a cut whose window opens before the fire is lit, and spans
    less than the flight, is refused naming both.
    """
    sample = Sample(burning_m2=(0.0, 33600.0), hottest_k=319.8, times_s=(0.0, 135.0))

    with raises(ValueError, match="no cell burns, .* spans 135 s, short of 150 s"):
        WrfOutput.from_sample(
            developing,
            sample   = sample,
            settings = Settings(crop=crop, plan={"duration_s": 150.0})
        )


def test_an_output_lacking_a_variable_is_refused_naming_it(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that an output lacking a variable the record keeps is refused,
    naming that variable, such as a simulation run without its smoke tracer.
    """
    with raises(KeyError, match="tr17_1"):
        WrfOutput.from_dataset(threat.dataset.drop_vars("tr17_1"), crop=crop)


def test_an_output_on_disk_keeps_every_field_unloaded(
    crop     : Crop,
    threat   : ParametricThreat,
    tmp_path : Path
):
    """
    Asserts that opening and cropping an output on disk leaves each field
    unread, which xarray reports through `Variable._in_memory`.
    """
    threat.dataset.to_netcdf(path := tmp_path / "wrfout.nc", engine="netcdf4")

    dataset = WrfOutput.from_path(path, crop=crop).dataset

    assert dataset.data_vars
    assert not any(dataset[name].variable._in_memory for name in dataset.data_vars)


def test_an_output_on_disk_reads_as_the_dataset_it_holds(
    crop     : Crop,
    output   : WrfOutput,
    threat   : ParametricThreat,
    tmp_path : Path
):
    """
    Asserts that an output opened from a NetCDF file reads as the same
    dataset the threat builds, cropped alike.
    """
    threat.dataset.to_netcdf(path := tmp_path / "wrfout.nc", engine="netcdf4")

    assert_identical(
        WrfOutput.from_path(path, crop=crop).dataset.load(),
        output.dataset
    )


@mark.parametrize(
    ("crop", "message"),
    [
        param(
            Crop(floor_m=5.0, levels=5, spacing_m=20.0),
            "cover 10 to .* short of the band from 5 to 85 m",
            id = "below-the-lowest-level"
        ),
        param(
            Crop(floor_m=20.0, levels=20, spacing_m=20.0),
            "short of the band from 20 to 400 m",
            id = "above-the-highest-level"
        ),
        param(
            Crop(floor_m=10.0, levels=12, spacing_m=20.001),
            "short of the band from 10 to 230.011 m",
            id = "a-centimeter-above-the-highest-level"
        )
    ]
)
def test_the_air_aloft_refuses_heights_its_levels_do_not_reach(
    crop    : Crop,
    message : str,
    threat  : ParametricThreat
):
    """
    Asserts that regridding onto fixed heights beginning below the lowest
    level, or ending above the highest the crop keeps, is refused rather
    than read as the value at the column's edge.
    """
    with raises(ValueError, match=message):
        WrfOutput.from_threat(threat, crop=crop).aloft


def test_destaggering_averages_each_pair_of_faces():
    """
    Asserts that a field destaggers onto the points between each pair of its
    faces, along the dimension named without its `_stag` suffix and carrying
    no coordinate there.
    """
    field = DataArray([0.0, 2.0, 6.0], coords={"x_stag": [0, 1, 2]}, dims="x_stag")

    averaged = destaggered(field, dimension="x_stag")

    assert averaged.dims == ("x",)
    assert averaged.values.tolist() == [1.0, 4.0]
    assert "x" not in averaged.coords


@mark.parametrize(
    "scale",
    [
        param(3.0, id="starting-above-the-floor"),
        param(0.25, id="ending-below-the-top")
    ]
)
def test_the_air_aloft_refuses_a_band_one_column_misses(
    crop   : Crop,
    scale  : float,
    threat : ParametricThreat
):
    """
    Asserts that the air aloft is refused where one column alone starts
    above the band or ends below it, every other column covering the band.
    """
    with raises(ValueError, match="short of the band"):
        WrfOutput.from_dataset(scaled(threat, scale=scale), crop=crop).aloft


@mark.parametrize(
    "level_m",
    [
        param(20.004, id="starting-two-millimeters-above-the-floor"),
        param(19.9998, id="ending-two-millimeters-below-the-top")
    ]
)
def test_the_air_aloft_refuses_a_band_missed_by_two_millimeters(level_m: float):
    """
    Asserts that the air aloft compares a column's extent to the band to
    the millimeter, refusing levels whose lowest center sits 2 mm above the
    floor or whose highest center sits 2 mm below the top.
    """
    threat = ParametricThreat(
        edge_m = 10.0,
        grid   = Grid(
            columns   = 20,
            level_m   = level_m,
            levels    = 12,
            rows      = 3,
            snapshots = 3
        )
    )
    crop = Crop(floor_m=10.0, levels=12, spacing_m=20.0)

    with raises(ValueError, match="short of the band"):
        WrfOutput.from_threat(threat, crop=crop).aloft


def test_each_wind_is_averaged_from_its_faces_onto_the_mass_points(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that `u`, `v`, and `w` take the mean of the two faces of each
    cell along the dimension the component is staggered on.
    """
    dataset = threat.dataset
    faces   = {
        "U" : ("west_east_stag", 21),
        "V" : ("south_north_stag", 4),
        "W" : ("bottom_top_stag", 13)
    }
    wind = WrfOutput.from_dataset(
        dataset.assign(
            {
                name: DataArray(
                    arange(size, dtype="float32"),
                    dims = dimension
                ).broadcast_like(
                    dataset[name]
                )
                for name, (dimension, size) in faces.items()
            }
        ),
        crop = crop
    ).wind.isel(Time=0)

    assert_allclose(wind.u.isel(bottom_top=0, south_north=0).values, arange(20) + 0.5)
    assert_allclose(wind.v.isel(bottom_top=0, west_east=0).values, arange(3) + 0.5)
    assert_allclose(wind.w.isel(south_north=0, west_east=0).values, arange(7) + 0.5)


def test_the_air_aloft_is_each_column_interpolated_onto_the_fixed_heights(
    crop   : Crop,
    output : WrfOutput,
    threat : ParametricThreat
):
    """
    Asserts that each field on the fixed heights is its column interpolated
    linearly in height, exact here, since the threat's air temperature in a
    column is linear in height and its smoke tracer constant along it.
    """
    aloft  = output.aloft
    column = threat.dataset.isel(bottom_top=0, south_north=0, west_east=3)

    assert aloft.temperature.dims == ("Time", "height", "south_north", "west_east")
    assert_allclose(
        aloft.temperature.isel(south_north=0, west_east=3).values,
        ((column.T + BASE_K) * exner(crop.heights)).values,
        rtol = 1e-5
    )
    assert_allclose(
        output.smoke.isel(south_north=0, west_east=3).values,
        column.tr17_1.broadcast_like(crop.heights).transpose("Time", "height").values,
        rtol = 1e-5
    )


def test_the_air_aloft_takes_a_band_beginning_on_the_level_it_reads():
    """
    Asserts that a band beginning at the lowest level's center is regridded,
    though that center lands a fraction of a micrometer above it in single
    precision on levels 7.2 m deep.
    """
    threat = ParametricThreat(
        edge_m = 10.0,
        grid   = Grid(
            columns   = 6,
            level_m   = 7.2,
            levels    = 4,
            rows      = 2,
            snapshots = 2
        )
    )
    crop = Crop(floor_m=3.6, levels=2, spacing_m=7.2)

    assert WrfOutput.from_threat(threat, crop=crop).aloft.sizes["height"] == 2


def test_the_air_aloft_takes_a_band_ending_on_the_levels_it_reads(
    threat: ParametricThreat
):
    """
    Asserts that a band beginning at the lowest level's center and ending
    at the highest's is regridded, though those centers land tens of
    micrometers off in single precision.
    """
    crop = Crop(floor_m=10.0, levels=12, spacing_m=20.0)

    assert WrfOutput.from_threat(threat, crop=crop).aloft.sizes["height"] == 12


@mark.parametrize(
    ("crop", "times", "sizes"),
    [
        param(
            Crop(floor_m=20.0, levels=5, spacing_m=20.0),
            [0.0, 15.0, 30.0],
            (3, 20, 21, 7, 8),
            id = "every-snapshot-and-column"
        ),
        param(
            Crop(
                floor_m   = 20.0,
                levels    = 5,
                spacing_m = 20.0,
                window_s  = (10.0, 40.0)
            ),
            [15.0, 30.0],
            (3, 20, 21, 7, 8),
            id = "a-window"
        ),
        param(
            Crop(
                floor_m   = 20.0,
                levels    = 5,
                spacing_m = 20.0,
                window_s  = (15.0, 30.0)
            ),
            [15.0, 30.0],
            (3, 20, 21, 7, 8),
            id = "a-window-whose-bounds-are-snapshots"
        ),
        param(
            Crop(
                floor_m   = 20.0,
                levels    = 5,
                spacing_m = 20.0,
                window_s  = (15.0, 15.0)
            ),
            [15.0],
            (3, 20, 21, 7, 8),
            id = "a-window-on-one-snapshot"
        ),
        param(
            Crop(
                floor_m   = 20.0,
                levels    = 5,
                spacing_m = 20.0,
                x_m       = (100.0, 300.0),
                y_m       = (0.0, 70.0)
            ),
            [0.0, 15.0, 30.0],
            (2, 6, 7, 7, 8),
            id = "a-neighborhood"
        )
    ]
)
def test_a_crop_keeps_the_window_the_neighborhood_and_the_band(
    crop   : Crop,
    sizes  : tuple[int, ...],
    threat : ParametricThreat,
    times  : list[float]
):
    """
    Asserts that a crop keeps the snapshots inside its window, the mass
    points inside its neighborhood with the staggered points bounding them,
    and the levels up to one past the first reaching the band's top, with
    the time as seconds since the simulation started.
    """
    dataset = WrfOutput.from_threat(threat, crop=crop).dataset

    assert dataset.Time.values.tolist() == times
    assert tuple(
        dataset.sizes[name]
        for name in (
            "south_north", "west_east", "west_east_stag",
            "bottom_top", "bottom_top_stag"
        )
    ) == sizes
    assert "XTIME" not in dataset


def test_the_air_lays_out_every_channel_finitely(output: WrfOutput):
    """
    Asserts that the air holds the winds, the smoke, the air temperature,
    its excess, and the distance to the front, every value finite even in
    the snapshot where nothing burns.
    """
    air = output.air

    assert air.names == ("u", "v", "w", "smoke", "temperature", "excess", "front")
    assert air.channels.shape == (7, 3, 5, 3, 20)
    assert isfinite(air.channels).all()


def test_the_air_temperature_is_the_potential_temperature_at_its_pressure(
    output : WrfOutput,
    threat : ParametricThreat
):
    """
    Asserts that the air temperature is `T` plus 300 K scaled by the Exner
    function of the pressure `P` plus `PB`, which the threat's base state
    sets to its dry adiabatic profile.
    """
    potential = threat.dataset.T.isel(bottom_top=slice(7)) + BASE_K

    assert_allclose(
        output.temperature.values,
        (potential * exner(threat.grid.heights.isel(bottom_top=slice(7)))).values,
        rtol = 1e-5
    )


@mark.parametrize(
    ("values", "held"),
    [
        param([0.0, 10.0], nullcontext(), id="reaching-each-bound"),
        param(
            [-0.5, 10.0],
            raises(
                ValueError,
                match = "`air` runs from -0.5 to 10 K, outside the range from 0 to 10 K"
            ),
            id = "under-the-floor"
        ),
        param(
            [0.0, 10.5],
            raises(
                ValueError,
                match = "`air` runs from 0 to 10.5 K, outside the range from 0 to 10 K"
            ),
            id = "past-the-ceiling"
        ),
        param(
            [5.0, nan],
            raises(ValueError, match="`air` runs from nan to nan K"),
            id = "missing-a-value"
        )
    ]
)
def test_a_field_checked_against_its_bounds_is_refused_past_either(
    held   : AbstractContextManager[object],
    values : list[float]
):
    """
    Asserts that a field reaching each of its bounds comes back as it went
    in, and one past either bound or missing a value is refused, the refusal
    naming the field, the values it holds, and its range.
    """
    field = DataArray(values, name="air")

    with held:
        assert checked(field, bounds=(0.0, 10.0), unit="K") is field


@mark.parametrize(
    "value",
    [
        param(-300.0, id="at-absolute-zero"),
        param(2000.0, id="past-a-crown-fire"),
        param(nan, id="missing")
    ]
)
def test_an_air_temperature_outside_its_range_is_refused(
    crop   : Crop,
    threat : ParametricThreat,
    value  : float
):
    """
    Asserts that an air temperature below the lowest recorded at the Earth's
    surface, past the figure a crown fire's peak exceeded, or missing is
    refused against the range `Ranges` declares.
    """
    data   = threat.dataset
    output = WrfOutput.from_dataset(data.assign(T=full_like(data.T, value)), crop=crop)

    with raises(
        ValueError,
        match = "`temperature` .* outside the range from 183.95 to 1603.15 K"
    ):
        output.temperature


@mark.parametrize(
    "scale",
    [param(-2.0, id="below-the-ground"), param(100.0, id="past-the-deepest-simulation")]
)
def test_a_height_outside_its_range_is_refused(
    crop   : Crop,
    scale  : float,
    threat : ParametricThreat
):
    """
    Asserts that a height below the ground or past the top of the deepest
    simulation is refused against the range `Ranges` declares.
    """
    data   = threat.dataset
    output = WrfOutput.from_dataset(data.assign(PH=data.PHB * scale), crop=crop)

    with raises(ValueError, match="`height` .* outside the range from 0 to 5000 m"):
        output.heights


def test_the_conversions_read_each_perturbation_and_the_ground(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that the air temperature reads the perturbation pressure beside
    the base state, and that the height reads the perturbation geopotential
    beside the base state, measured from the ground's own.
    """
    data   = threat.dataset
    output = WrfOutput.from_dataset(
        data.assign(
            P   = data.P + 500.0,
            PH  = data.PHB * 0.5,
            PHB = data.PHB + GRAVITY_M_S2 * 100.0
        ),
        crop = crop
    )
    levels   = data.isel(bottom_top=slice(output.dataset.sizes["bottom_top"]))
    exponent = DRY_AIR_J_KG_K / HEAT_CAPACITY_J_KG_K

    assert_allclose(
        output.temperature.values,
        ((levels.T + BASE_K) * ((levels.PB + 500.0) / REFERENCE_PA) ** exponent).values,
        rtol = 1e-5
    )
    assert_allclose(
        output.heights.isel(Time=0, south_north=0, west_east=0).values,
        1.5 * threat.grid.heights.isel(bottom_top=levels.bottom_top).values,
        rtol = 1e-4
    )


@mark.parametrize(
    "values",
    [
        param([0.0, 1.0], id="a-tracer-at-rest"),
        param([-8.9e-8, 11905.0], id="undershooting-zero-as-the-sample-does"),
        param([-1e36, 1e36], id="short-of-the-fill-value-either-way")
    ]
)
def test_a_field_with_no_range_takes_any_value_short_of_the_fill_value(
    values: list[float]
):
    """
    Asserts that a field holding no range comes back as it went in whatever
    it holds short of the netCDF fill value, so no physical bound applies
    to it.
    """
    field = DataArray(array(values, dtype="float32"), name="tr17_1")

    assert unfilled(field) is field


@mark.parametrize(
    ("values", "reading"),
    [
        param([5.0, nan], "nan to nan", id="missing-a-value"),
        param([nan, nan], "nan to nan", id="missing-every-value"),
        param([5.0, FILL_VALUE], r"5 to 9\.96921e\+36", id="at-the-fill-value"),
        param([-FILL_VALUE, 5.0], r"-9\.96921e\+36 to 5", id="at-its-negative"),
        param([5.0, inf], "5 to inf", id="infinite"),
        param([-inf, 5.0], "-inf to 5", id="negatively-infinite")
    ]
)
def test_a_field_with_no_range_is_refused_naming_the_values_it_holds(
    reading : str,
    values  : list[float]
):
    """
    Asserts that a field holding a missing value, or a value at or past the
    fill value netCDF writes by default in single precision, is refused,
    naming the field and the values it holds.
    """
    with raises(
        ValueError,
        match = (
            f"`tr17_1` runs from {reading}, holding a missing value or reaching "
            r"the netCDF fill value of 9\.96921e\+36"
        )
    ):
        unfilled(DataArray(array(values, dtype="float32"), name="tr17_1"))


@mark.parametrize(
    "value",
    [param(nan, id="missing"), param(FILL_VALUE, id="at-the-fill-value")]
)
@mark.parametrize(
    ("variable", "reader"),
    [
        param("GRNHFX", "air", id="the-air-from-the-ground-heat-flux"),
        param("GRNHFX", "front", id="the-front-from-the-ground-heat-flux"),
        param("tr17_1", "air", id="the-air-from-the-smoke-tracer"),
        param("tr17_1", "smoke", id="the-smoke-from-its-tracer")
    ]
)
def test_a_bad_value_in_a_field_with_no_range_is_refused_where_it_is_read(
    crop     : Crop,
    reader   : str,
    threat   : ParametricThreat,
    value    : float,
    variable : str
):
    """
    Asserts that one cell of the ground heat flux or the smoke tracer
    holding a missing or fill value is refused wherever the record reads
    that field, naming the variable, rather than reading as no fire or
    reaching the air.
    """
    data = threat.dataset
    bad  = data[variable].copy()
    bad[{"Time": -1, "south_north": 0, "west_east": 0}] = value
    output = WrfOutput.from_dataset(data.assign({variable: bad}), crop=crop)

    with raises(ValueError, match=f"`{variable}` runs from "):
        getattr(output, reader)


@mark.parametrize("offset", [param(-200.0, id="westward"), param(200.0, id="eastward")])
@mark.parametrize("variable", ["U", "V", "W"])
def test_a_wind_outside_its_range_is_refused_naming_its_axis(
    crop     : Crop,
    offset   : float,
    threat   : ParametricThreat,
    variable : str
):
    """
    Asserts that a wind past the world-record gust, either way along any
    axis, is refused against the range `Ranges` declares, the refusal naming
    the component.
    """
    data   = threat.dataset
    output = WrfOutput.from_dataset(
        data.assign({variable: data[variable] + offset}),
        crop = crop
    )

    with raises(
        ValueError,
        match = f"`{variable.lower()}` .* outside the range from -113.3 to 113.3 m/s"
    ):
        output.wind


def test_the_default_settings_lay_out_the_default_threat():
    """
    Asserts that the fixed heights the default crop lays out sit inside the
    levels of the default threat, so the settings a fresh clone reads lay
    out every channel finitely.
    """
    air = WrfOutput.from_settings(Settings()).air

    assert air.channels.shape == (7, 9, 25, 50, 100)
    assert isfinite(air.channels).all()


def test_the_distances_measure_each_cell_to_the_nearest_burning_one():
    """
    Asserts that each cell's distance is measured to the nearest burning
    cell along the spacing of the rows and the columns, and that every
    distance is infinite where no cell burns.
    """
    burning       = zeros((2, 2), dtype=bool)
    burning[0, 0] = True

    assert_allclose(
        distances(burning, sampling=(2.0, 3.0)),
        array([[0.0, 3.0], [2.0, sqrt(13.0)]])
    )
    assert (distances(zeros((2, 2), dtype=bool), sampling=(2.0, 3.0)) == inf).all()


def test_the_excess_is_the_heat_behind_the_front_over_the_air_ahead(
    crop   : Crop,
    output : WrfOutput,
    threat : ParametricThreat
):
    """
    Asserts that the ambient profile at each height is the air ahead of the
    front, which covers most of the domain, so the excess is nothing far
    ahead of the front and the anomaly's share of the air temperature far
    behind it.
    """
    excess = output.excess.isel(Time=-1, south_north=0)

    assert_allclose(
        output.ambient.isel(Time=-1).values,
        (BASE_K * exner(crop.heights)).values,
        rtol = 1e-5
    )
    assert_allclose(excess.isel(west_east=-1).values, 0.0, atol=1e-3)
    assert_allclose(
        excess.isel(west_east=0).values,
        (threat.anomaly_k * exner(crop.heights)).values,
        rtol = 1e-3
    )


def test_the_extremes_read_a_missing_value_as_nan():
    """
    Asserts that the extremes of a field are its lowest and highest values,
    and that a single missing value turns both into NaN.
    """
    assert extremes(DataArray([3.0, 1.0, 2.0])) == (1.0, 3.0)
    assert all(map(isnan, extremes(DataArray([3.0, nan, 2.0]))))


def test_the_front_and_the_ambient_read_the_columns_the_crop_keeps(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that a neighborhood ahead of the front holds no burning point,
    so its distance to the front is infinite, and that one wholly behind the
    front takes the hot air as its ambient profile, leaving no excess.
    """
    ahead  = crop.model_copy(update={"x_m": (500.0, 800.0)})
    behind = crop.model_copy(update={"x_m": (0.0, 200.0)})

    assert (WrfOutput.from_threat(threat, crop=ahead).front.values == inf).all()
    assert_allclose(
        WrfOutput.from_threat(threat, crop=behind).excess.isel(Time=-1).values,
        0.0,
        atol = 1e-3
    )


def test_the_front_and_the_points_read_each_axis_s_own_spacing(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that an output whose rows lie 80 m apart and whose columns lie
    40 m apart places its points and measures the front along each axis by
    that axis's spacing, and the area burning.
    """
    data    = threat.dataset
    burning = zeros_like(data.GRNHFX)
    burning[{"south_north": 0, "west_east": 0}] = 1.0

    output = WrfOutput.from_dataset(
        data.assign(GRNHFX=burning).assign_attrs(DY=80.0),
        crop = crop
    )

    assert output.dataset.south_north.values.tolist() == [40.0, 120.0, 200.0]
    assert_allclose(output.front.isel(Time=0, south_north=1, west_east=0), 80.0)
    assert_allclose(output.front.isel(Time=0, south_north=0, west_east=1), 40.0)
    assert_allclose(output.plume.burning_m2, 3200.0)


def test_the_front_is_each_column_s_distance_to_the_nearest_burning_one(
    output: WrfOutput
):
    """
    Asserts that every distance to the front is infinite in a snapshot where
    nothing burns, and otherwise each column's distance to the easternmost
    burning one, which the front reaches at 300 m by the last snapshot.
    """
    front = output.front.isel(south_north=1)

    assert (front.isel(Time=0).values == inf).all()
    assert_allclose(
        front.isel(Time=-1).values,
        [0.0] * 8 + [40.0 * step for step in range(1, 13)]
    )


def test_the_front_reads_only_a_flux_above_zero_as_burning(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that a cell whose ground heat flux sits barely above zero burns,
    while a cell at zero or below it does not.
    """
    data = threat.dataset
    flux = zeros_like(data.GRNHFX)
    flux[{"south_north": 0, "west_east": 0}] = 0.001
    flux[{"south_north": 0, "west_east": 1}] = -1.0

    front = WrfOutput.from_dataset(
        data.assign(GRNHFX=flux),
        crop = crop
    ).front.isel(Time=0)

    assert_allclose(front.isel(south_north=0, west_east=0), 0.0)
    assert_allclose(front.isel(south_north=0, west_east=1), 40.0)
    assert_allclose(front.isel(south_north=1, west_east=1), 40.0 * sqrt(2))


def test_the_height_above_the_ground_is_each_level_s_center(
    output : WrfOutput,
    threat : ParametricThreat
):
    """
    Asserts that the height of each mass point is the middle of the two
    staggered levels around it, read from the geopotential above the
    ground's own.
    """
    expected = threat.grid.heights.isel(bottom_top=slice(7)).values

    assert_allclose(
        output.heights.isel(Time=0, south_north=0, west_east=0).values,
        expected,
        rtol = 1e-6
    )
    assert_allclose(
        output.heights.isel(Time=-1, south_north=-1, west_east=-1).values,
        expected,
        rtol = 1e-6
    )


def test_the_plume_measures_the_fire_and_the_air_above_it(output: WrfOutput):
    """
    Asserts that the plume measures the area burning, the excess above
    a burning cell, the excess in air the smoke has not reached, and the
    hottest air, in each snapshot, leaving the excess above the fire NaN
    before anything burns.
    """
    plume = output.plume

    assert_allclose(plume.burning_m2, [0.0, 19200.0, 38400.0])
    assert_allclose(plume.convected_k, [0.0, 0.0, 0.0])
    assert isnan(plume.heated_k[0])
    assert_allclose(plume.heated_k[1:], 20.0, atol=0.02)
    assert_allclose(plume.hottest_k, [300.2, 319.8, 319.8], atol=0.05)


def test_the_plume_reads_the_convection_in_air_the_smoke_has_not_reached(
    crop   : Crop,
    threat : ParametricThreat
):
    """
    Asserts that once the smoke tracer reaches no air, the excess the
    ambient convection is read from covers the air above the fire as well.
    """
    data  = threat.dataset
    plume = WrfOutput.from_dataset(
        data.assign(tr17_1=zeros_like(data.tr17_1)),
        crop = crop
    ).plume

    assert_allclose(plume.convected_k[1:], plume.heated_k[1:])


@mark.parametrize(
    ("bounds", "mass", "staggered"),
    [
        param((50.0, 110.0), slice(1, 3), slice(1, 4), id="a-span"),
        param((60.0, 100.0), slice(1, 3), slice(1, 4), id="bounds-on-the-points"),
        param((60.0, 60.0), slice(1, 2), slice(1, 3), id="bounds-on-one-point"),
        param(None, slice(0, 4), slice(0, 5), id="no-span")
    ]
)
def test_a_span_keeps_the_mass_points_inside_it_and_the_faces_bounding_them(
    bounds    : Span,
    mass      : slice,
    staggered : slice
):
    """
    Asserts that a span keeps each mass point whose coordinate falls inside
    its bounds and the staggered points either side of them, and every point
    where it names no bounds.
    """
    coordinate = Dataset(coords={"x": [20.0, 60.0, 100.0, 140.0]}).x

    assert spans(coordinate, bounds=bounds) == {"x": mass, "x_stag": staggered}


def test_the_settings_choose_the_simulation_or_the_threat(
    crop     : Crop,
    output   : WrfOutput,
    threat   : ParametricThreat,
    tmp_path : Path
):
    """
    Asserts that settings naming no simulation read the threat, and that
    settings naming one read the file it names.
    """
    threat.dataset.assign(T=threat.dataset.T + 1).to_netcdf(
        path := tmp_path / "wrfout.nc",
        engine = "netcdf4"
    )

    read = WrfOutput.from_settings(Settings(crop=crop, simulation=path, threat=threat))

    assert_identical(
        WrfOutput.from_settings(Settings(crop=crop, threat=threat)).dataset,
        output.dataset
    )
    assert_allclose(read.dataset.T.values, output.dataset.T.values + 1)


def test_the_settings_name_a_simulation_relative_to_the_clone(
    crop        : Crop,
    monkeypatch : MonkeyPatch,
    output      : WrfOutput,
    threat      : ParametricThreat,
    tmp_path    : Path
):
    """
    Asserts that a simulation named by a relative path is read under the
    clone the package was installed from.
    """
    (tmp_path / "data").mkdir()
    threat.dataset.assign(T=threat.dataset.T + 1).to_netcdf(
        tmp_path / "data/wrfout.nc",
        engine = "netcdf4"
    )
    monkeypatch.setattr(
        Checkout,
        "installed",
        classmethod(lambda _: Checkout(root=tmp_path))
    )

    read = WrfOutput.from_settings(
        Settings(crop=crop, simulation=Path("data/wrfout.nc"), threat=threat)
    )

    assert_allclose(read.dataset.T.values, output.dataset.T.values + 1)


def test_the_settings_naming_a_missing_simulation_raise(
    crop     : Crop,
    threat   : ParametricThreat,
    tmp_path : Path
):
    """
    Asserts that settings naming a file the clone does not hold raise the
    error naming it, rather than reading the threat.
    """
    settings = Settings(crop=crop, simulation=tmp_path / "absent.nc", threat=threat)

    with raises(FileNotFoundError, match="absent.nc"):
        WrfOutput.from_settings(settings)


def test_the_times_keep_no_attribute_of_xtime(crop: Crop, threat: ParametricThreat):
    """
    Asserts that the times in seconds a crop places carry none of the
    attributes describing `XTIME` in minutes.
    """
    data   = threat.dataset
    output = WrfOutput.from_dataset(
        data.assign(
            XTIME = data.XTIME.assign_attrs(
                description = "minutes since simulation start"
            )
        ),
        crop = crop
    )

    assert output.dataset.Time.attrs == {}


def test_the_winds_read_no_air_temperature(crop: Crop, threat: ParametricThreat):
    """
    Asserts that the winds come back from an output whose air temperature is
    missing, so a bad temperature raises only where the temperature is read.
    """
    data   = threat.dataset
    output = WrfOutput.from_dataset(data.assign(T=full_like(data.T, nan)), crop=crop)

    assert output.wind.sizes["bottom_top"] == 7
