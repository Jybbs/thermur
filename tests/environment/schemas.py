"""
Pins the records the fire data is read through:

- The heights a `Crop` lays out
- The points and sizes of the `Grid` a threat lays its dataset out on
- The dataset a `ParametricThreat` builds under WRF's names, and the reach
  its validator holds the front to
- The address of the `Sample`, the request it sends, the fetch that checks
  and moves its file into place, and the checks it holds a window's
  figures to
"""

from contextlib         import AbstractContextManager, nullcontext
from hashlib            import sha256
from importlib.metadata import version
from numpy              import nan
from numpy.testing      import assert_allclose
from pathlib            import Path
from pydantic           import ValidationError
from pytest             import MonkeyPatch, mark, param, raises
from tempfile           import NamedTemporaryFile
from unittest.mock      import Mock
from urllib.error       import URLError
from urllib.request     import Request, urlopen
from xarray             import Dataset

from thermur.environment.schemas import (
    BASE_K,
    Crop,
    DRY_AIR_J_KG_K,
    GRAVITY_M_S2,
    Grid,
    HEAT_CAPACITY_J_KG_K,
    ParametricThreat,
    REFERENCE_PA,
    Sample
)
from thermur.repo.checkout import Checkout


def measured(**figures: list[float]) -> Dataset:
    """
    Lays out the figures `WrfOutput.plume` measures over three snapshots
    15 s apart, each figure `figures` names replacing the one a developed
    plume holds.
    """
    plume = {
        "Time"        : [15.0, 30.0, 45.0],
        "burning_m2"  : [100.0, 200.0, 3000.0],
        "convected_k" : [1.0, 2.0, 1.5],
        "heated_k"    : [5.0, 6.0, 7.0],
        "hottest_k"   : [310.0, 312.04, 311.0]
    } | figures
    return Dataset(
        {name: ("Time", values) for name, values in plume.items() if name != "Time"},
        coords = {"Time": plume["Time"]}
    )


def pinned() -> Sample:
    """
    Builds the sample pinning the figures `measured` lays out by default.
    """
    return Sample(
        burning_m2 = (100.0, 3000.0),
        hottest_k  = 312.0,
        times_s    = (15.0, 30.0, 45.0)
    )


def served(data: bytes, folder: Path, *, digest: str | None = None) -> Sample:
    """
    Writes `data` where the sample's address points once `folder` stands in
    for the bucket.

    Returns:
        The sample pinning `digest`, or the digest of `data` where none is
        given, with `folder` as its host.
    """
    sample = Sample(digest=digest or sha256(data).hexdigest(), host=folder.as_uri())
    (path := Path.from_uri(sample.address)).parent.mkdir(parents=True)
    path.write_bytes(data)
    return sample


def test_a_crop_lays_its_heights_out_evenly_from_its_floor():
    """
    Asserts that a crop's fixed heights start at its floor and step by its
    spacing, one per level, carried as the `height` coordinate.
    """
    heights = Crop(floor_m=10.0, levels=3, spacing_m=5.0).heights

    assert heights.dims == ("height",)
    assert heights.values.tolist() == [10.0, 15.0, 20.0]
    assert heights.height.values.tolist() == [10.0, 15.0, 20.0]


def test_a_fetch_downloads_into_a_temporary_file_beside_its_target(
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that a fetch writes its download into a temporary file in the
    folder the file lands in.
    """
    spy = Mock(wraps=NamedTemporaryFile)
    monkeypatch.setattr("thermur.environment.schemas.NamedTemporaryFile", spy)

    served(b"CDF", tmp_path / "bucket").fetch(tmp_path / "clone")

    spy.assert_called_once_with(dir=tmp_path / "clone/data/samples")


def test_a_fetch_moves_a_file_matching_its_digest_into_place(tmp_path: Path):
    """
    Asserts that a fetch downloads the file, moves it to the sample's path
    under the clone, leaves nothing else in its folder, and returns that
    path.
    """
    fetched = served(b"CDF", tmp_path / "bucket").fetch(tmp_path / "clone")

    assert fetched == tmp_path / "clone/data/samples/wrf_sample.nc"
    assert fetched.read_bytes() == b"CDF"
    assert list(fetched.parent.iterdir()) == [fetched]


def test_a_fetch_moves_nothing_from_a_file_with_another_digest(tmp_path: Path):
    """
    Asserts that a fetch whose file hashes to another digest than the pinned
    one raises naming both, leaving the folder empty.
    """
    sample = served(b"CDF", tmp_path / "bucket", digest="0" * 64)

    with raises(
        ValueError,
        match = (
            f"hashes to `{sha256(b'CDF').hexdigest()}` rather than the pinned "
            f"`{'0' * 64}`"
        )
    ):
        sample.fetch(tmp_path / "clone")

    assert list((tmp_path / "clone/data/samples").iterdir()) == []


def test_a_fetch_waits_on_its_download_for_at_most_its_timeout(
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that a fetch passes its timeout to the download, and that a
    download stalling past it raises `TimeoutError`, leaving the folder
    empty.
    """
    waits: list[float] = []

    def stalled(request: Request, *, timeout: float):
        """
        Records the timeout a download is given and raises as one stalling
        past it does.
        """
        waits.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr("thermur.environment.schemas.urlopen", stalled)

    with raises(TimeoutError):
        Sample(timeout_s=5.0).fetch(tmp_path / "clone")

    assert waits == [5.0]
    assert list((tmp_path / "clone/data/samples").iterdir()) == []


def test_a_fetch_whose_download_fails_moves_nothing(tmp_path: Path):
    """
    Asserts that a fetch whose download fails raises the error the download
    raised, leaving the folder empty.
    """
    with raises(URLError):
        Sample(host=(tmp_path / "bucket").as_uri()).fetch(tmp_path / "clone")

    assert list((tmp_path / "clone/data/samples").iterdir()) == []


@mark.parametrize(
    ("path", "status"),
    [
        param(Sample().path, 0, id="the-fetched-output"),
        param(Path("docs/pages/data/index.md"), 1, id="the-data-page"),
        param(Path(".mise/tasks/data/fetch"), 1, id="the-data-task")
    ]
)
def test_git_ignores_the_fetched_output_and_no_other_data_folder(
    checkout : Checkout,
    path     : Path,
    status   : int
):
    """
    Asserts that git ignores the file a fetch writes under the clone's
    `data/` folder, and no file under a `data` folder deeper in the clone.
    """
    assert checkout.run("git", "check-ignore", path, check=False).returncode == status


def test_a_grid_places_each_point_from_its_spacings():
    """
    Asserts that a grid's mass points sit midway between its staggered ones,
    its staggered levels start at the ground, and its snapshots start at
    zero, each along the WRF dimension it lies on.
    """
    grid = Grid(
        columns    = 3,
        interval_s = 10.0,
        level_m    = 4.0,
        levels     = 2,
        snapshots  = 3,
        spacing_m  = 2.0
    )

    assert (
        grid.centers.dims,
        grid.centers.values.tolist()
    ) == (("west_east",), [1.0, 3.0, 5.0])
    assert (grid.depths.dims, grid.depths.values.tolist()) == (
        ("bottom_top_stag",),
        [0.0, 4.0, 8.0]
    )
    assert (
        grid.heights.dims,
        grid.heights.values.tolist()
    ) == (("bottom_top",), [2.0, 6.0])
    assert (
        grid.seconds.dims,
        grid.seconds.values.tolist()
    ) == (("Time",), [0.0, 10.0, 20.0])
    assert (grid.length_m, grid.span_s) == (6.0, 20.0)


def test_a_grid_sizes_each_staggered_dimension_one_past_its_mass_points():
    """
    Asserts that a grid sizes every dimension WRF names, each staggered
    one holding one point more than the mass points it bounds, and lays its
    zeros out over the dimensions named in their order.
    """
    grid = Grid(
        columns   = 4,
        levels    = 3,
        rows      = 2,
        snapshots = 5
    )

    assert grid.sizes == {
        "Time"             : 5,
        "bottom_top"       : 3,
        "bottom_top_stag"  : 4,
        "south_north"      : 2,
        "south_north_stag" : 3,
        "west_east"        : 4,
        "west_east_stag"   : 5
    }
    assert grid.zeros("west_east_stag", "Time").shape == (5, 5)


@mark.parametrize(
    ("sample", "address"),
    [
        param(
            Sample(),
            "https://data.thermur.fyi/samples/"
            "620c829bc086d0e014290d7f4a31bb7cfd20ea5c646aca4a61df516342176c09/"
            "wrf_sample.nc",
            id = "the-pinned-sample"
        ),
        param(
            Sample(digest="a" * 64, host="https://bucket.test"),
            f"https://bucket.test/samples/{'a' * 64}/wrf_sample.nc",
            id = "another-digest-on-another-host"
        )
    ]
)
def test_a_sample_addresses_its_file_under_a_key_carrying_its_digest(
    address : str,
    sample  : Sample
):
    """
    Asserts that the sample's address names its host and then the key the
    file is published under, `samples/`, the digest, and the file, so each
    digest takes an address of its own.
    """
    assert sample.address == address


def test_a_sample_attributes_its_file_to_the_dataset_it_is_cut_from():
    """
    Asserts that the attributes a written sample carries name its license
    and, as its source, the simulation and the credit of the dataset the
    sample holds, crediting no creator the dataset does not name.
    """
    assert Sample(
        dataset    = "Test Dataset by A. Author, https://doi.org/10.0000/test",
        license    = "CC0 1.0",
        simulation = "wrfout_W3F1R0"
    ).attributes == {
        "license" : "CC0 1.0",
        "source"  : (
            "Cut from wrfout_W3F1R0 of the Test Dataset by A. Author, "
            "https://doi.org/10.0000/test"
        )
    }


@mark.parametrize(
    ("columns", "speed_m_s", "held"),
    [
        param(50, 7.8, nullcontext(), id="reaching-the-column-of-an-even-count"),
        param(
            50,
            7.81,
            raises(ValidationError, match="reaches 981 m .* domain 2000 m long"),
            id = "passing-the-column-of-an-even-count"
        ),
        param(51, 8.2, nullcontext(), id="reaching-the-column-of-an-odd-count"),
        param(
            51,
            8.21,
            raises(ValidationError, match="reaches 1021 m .* domain 2040 m long"),
            id = "passing-the-column-of-an-odd-count"
        )
    ]
)
def test_a_threat_holds_its_front_short_of_the_column_a_median_reads(
    columns   : int,
    speed_m_s : float,
    held      : AbstractContextManager[object]
):
    """
    Asserts that a threat whose front and five times its `edge_m` ahead
    of it reach the center of the first column a median reads by the last
    snapshot validates, and one passing it is refused, on a domain the front
    crosses for 100 s behind an edge of 40 m, that column being the western
    of the two middle columns of an even count.
    """
    with held:
        ParametricThreat(
            grid = Grid(
                columns    = columns,
                interval_s = 10.0,
                snapshots  = 11,
                spacing_m  = 40.0
            ),
            speed_m_s = speed_m_s
        )


@mark.parametrize(
    ("record", "field", "value"),
    [
        param(Crop, "floor_m", 0.0, id="crop-floor"),
        param(Crop, "levels", 0, id="crop-levels"),
        param(Crop, "spacing_m", 0.0, id="crop-spacing"),
        param(Grid, "columns", 0, id="grid-columns"),
        param(Grid, "interval_s", 0.0, id="grid-interval"),
        param(Grid, "level_m", 0.0, id="grid-level-depth"),
        param(Grid, "levels", 0, id="grid-levels"),
        param(Grid, "rows", 0, id="grid-rows"),
        param(Grid, "snapshots", 0, id="grid-snapshots"),
        param(Grid, "spacing_m", 0.0, id="grid-spacing"),
        param(ParametricThreat, "anomaly_k", 0.0, id="threat-anomaly"),
        param(ParametricThreat, "edge_m", 0.0, id="threat-edge"),
        param(ParametricThreat, "speed_m_s", -1.0, id="threat-speed"),
        param(Sample, "timeout_s", 0.0, id="sample-timeout")
    ]
)
def test_a_record_refuses_a_value_past_a_field_s_bound(
    field  : str,
    record : type[Crop | Grid | ParametricThreat | Sample],
    value  : float
):
    """
    Asserts that each bounded field of the fire data's records refuses the
    first value past its bound, naming the field.
    """
    with raises(ValidationError, match=field):
        record(**{field: value})


def test_a_sample_refuses_a_window_naming_every_figure_it_misses():
    """
    Asserts that a window missing several figures is refused once, naming
    each of them.
    """
    with raises(ValueError, match="no cell burns, and burns 0 and 3,000 m²"):
        pinned().check(measured(burning_m2=[0.0, 200.0, 3000.0]), duration_s=30.0)


def test_a_sample_requests_its_file_naming_the_package_alone():
    """
    Asserts that the request for the file goes to the sample's address
    and names the package and its installed version as the client, the one
    header it sends.
    """
    request = Sample().request

    assert request.full_url == Sample().address
    assert request.header_items() == [("User-agent", f"thermur/{version('thermur')}")]


@mark.parametrize(
    "field",
    [
        param({"digest": "6f97"}, id="a-short-digest"),
        param({"digest": "E" * 64}, id="an-uppercase-digest")
    ]
)
def test_a_sample_refuses_a_pin_outside_its_form(field: dict[str, str]):
    """
    Asserts that a digest other than 64 lowercase hexadecimal digits is
    refused.
    """
    with raises(ValidationError):
        Sample(**field)


def test_a_sample_takes_a_window_holding_its_figures():
    """
    Asserts that a window falling at the pinned times, burning the pinned
    areas, and reaching the pinned hottest air to a tenth of a kelvin,
    with the air above the fire warmer than the ambient convection in every
    snapshot, passes the check.
    """
    pinned().check(measured(), duration_s=30.0)


def test_a_second_fetch_replaces_the_file_afresh(tmp_path: Path):
    """
    Asserts that fetching again over an earlier fetch replaces the file, so
    a file edited since reads as the bucket serves it.
    """
    sample = served(b"CDF", tmp_path / "bucket")
    sample.fetch(tmp_path / "clone").write_bytes(b"edited")

    assert sample.fetch(tmp_path / "clone").read_bytes() == b"CDF"


def test_a_still_threat_burns_nothing():
    """
    Asserts that a front advancing at no speed validates, leaving every cell
    of every snapshot unburned.
    """
    assert ParametricThreat(speed_m_s=0.0).dataset.GRNHFX.max() == 0


def test_a_threat_burns_a_cell_the_moment_its_front_reaches_the_center():
    """
    Asserts that a threat's ground heat flux shows a cell burning at the
    snapshot its front reaches the cell's center, and the next cell not yet.
    """
    threat  = ParametricThreat(speed_m_s=11.5, grid=Grid(interval_s=10.0, snapshots=13))
    burning = threat.dataset.GRNHFX.isel(Time=-1, south_north=0).values

    assert (burning[34], burning[35]) == (1.0, 0.0)


def test_a_threat_lays_its_front_out_under_wrf_s_names():
    """
    Asserts that the threat's dataset holds each variable a WRF-SFIRE output
    holds that the fire data reads, on the dimensions WRF writes it on and
    in single precision, with the grid's spacing as its `DX` and `DY`.
    """
    dataset = ParametricThreat().dataset

    assert {name: variable.dims for name, variable in dataset.data_vars.items()} == {
        "GRNHFX" : ("Time", "south_north", "west_east"),
        "P"      : ("Time", "bottom_top", "south_north", "west_east"),
        "PB"     : ("Time", "bottom_top", "south_north", "west_east"),
        "PH"     : ("Time", "bottom_top_stag", "south_north", "west_east"),
        "PHB"    : ("Time", "bottom_top_stag", "south_north", "west_east"),
        "T"      : ("Time", "bottom_top", "south_north", "west_east"),
        "U"      : ("Time", "bottom_top", "south_north", "west_east_stag"),
        "V"      : ("Time", "bottom_top", "south_north_stag", "west_east"),
        "W"      : ("Time", "bottom_top_stag", "south_north", "west_east"),
        "XTIME"  : ("Time",),
        "tr17_1" : ("Time", "bottom_top", "south_north", "west_east")
    }
    assert {dataset[name].dtype.name for name in dataset.data_vars} == {"float32"}
    assert dataset.attrs == {"DX": 40.0, "DY": 40.0}


def test_a_threat_s_base_state_cools_at_the_dry_adiabatic_rate():
    """
    Asserts that the threat's base-state pressure is the one at which air
    holding `BASE_K` of potential temperature cools by `g / c_p` per meter,
    and that its base-state geopotential is `g` times each staggered level's
    depth.
    """
    threat  = ParametricThreat()
    dataset = threat.dataset.isel(Time=0, south_north=0, west_east=0)

    assert_allclose(
        BASE_K * (dataset.PB / REFERENCE_PA) ** (DRY_AIR_J_KG_K / HEAT_CAPACITY_J_KG_K),
        BASE_K - GRAVITY_M_S2 * threat.grid.heights.values / HEAT_CAPACITY_J_KG_K,
        rtol = 1e-6
    )
    assert_allclose(
        dataset.PHB.values,
        GRAVITY_M_S2 * threat.grid.depths.values,
        rtol = 1e-6
    )


@mark.parametrize(
    ("figures", "duration_s", "message"),
    [
        param(
            {"Time": [15.0, 30.0, 60.0]},
            30.0,
            r"falls at \[15.0, 30.0, 60.0\] s rather than at \[15.0, 30.0, 45.0\] s",
            id = "at-other-times"
        ),
        param(
            {},
            60.0,
            "spans 30 s, short of 60 s",
            id = "too-short"
        ),
        param(
            {"burning_m2": [100.0, 0.0, 3000.0]},
            30.0,
            "holds a snapshot in which no cell burns",
            id = "burning-out"
        ),
        param(
            {"burning_m2": [100.0, 200.0, 4000.0]},
            30.0,
            "burns 100 and 4,000 m² rather than the pinned 100 and 3,000 m²",
            id = "burning-another-area"
        ),
        param(
            {"heated_k": [5.0, 2.0, 7.0]},
            30.0,
            "exceeds the ambient profile by no more than the 2.0 K",
            id = "matched-by-the-convection"
        ),
        param(
            {"convected_k": [nan, nan, nan]},
            30.0,
            "holds no air the smoke has not reached",
            id = "with-no-ambient-air"
        ),
        param(
            {"hottest_k": [310.0, 312.05, 311.0]},
            30.0,
            "reaches 312.1 K rather than the pinned 312 K",
            id = "reaching-other-air"
        )
    ]
)
def test_a_sample_refuses_a_window_missing_a_figure_naming_it(
    duration_s : float,
    figures    : dict[str, list[float]],
    message    : str
):
    """
    Asserts that a window missing any one figure the sample holds it to is
    refused with a message naming that figure.
    """
    with raises(ValueError, match=message):
        pinned().check(measured(**figures), duration_s=duration_s)


def test_a_threat_warms_and_burns_behind_its_front_alone():
    """
    Asserts that the threat's perturbation potential temperature reaches its
    anomaly far behind the front and nothing far ahead of it, as its smoke
    tracer reaches one and nothing, and that the ground burns on each cell
    whose center the front has reached and on no other.
    """
    threat  = ParametricThreat(anomaly_k=10.0, edge_m=1.0, speed_m_s=10.0)
    dataset = threat.dataset.isel(Time=-1, bottom_top=0, south_north=0)
    front   = threat.speed_m_s * threat.grid.span_s
    centers = threat.grid.centers.values

    assert_allclose(dataset.T.values[centers < front - 40], 10.0)
    assert_allclose(dataset.T.values[centers > front + 40], 0.0, atol=1e-6)
    assert_allclose(dataset.tr17_1.values[centers < front - 40], 1.0)
    assert (dataset.GRNHFX.values == (centers <= front)).all()
    assert threat.dataset.XTIME.values.tolist() == [
        seconds / 60 for seconds in threat.grid.seconds.values.tolist()
    ]


@mark.network
def test_the_bucket_serves_the_pinned_file_at_its_address():
    """
    Asserts that the bucket answers the request for the file at the sample's
    address with the file's pinned size, read from the response's headers
    rather than from the file, whose digest the fetch checks.
    """
    request        = Sample().request
    request.method = "HEAD"

    with urlopen(request, timeout=Sample().timeout_s) as response:
        assert response.status == 200
        assert int(response.headers["Content-Length"]) == 455334719
