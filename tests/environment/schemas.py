"""
Pins the records the fire data is read through:

- The heights a `Crop` lays out
- The points and sizes of the `Grid` a threat lays its dataset out on
- The dataset a `ParametricThreat` builds under WRF's names, and the reach
  its validator holds the front to
- The address of the `Sample`, the request it sends, and the fetch that
  checks and extracts its archive
"""

from contextlib         import AbstractContextManager, nullcontext
from hashlib            import sha256
from importlib.metadata import version
from io                 import BytesIO
from numpy.testing      import assert_allclose
from pathlib            import Path
from pydantic           import ValidationError
from pytest             import MonkeyPatch, mark, param, raises
from tarfile            import CHRTYPE, FilterError, SYMTYPE, TarFile, TarInfo
from tempfile           import TemporaryFile
from unittest.mock      import Mock
from urllib.error       import HTTPError, URLError
from urllib.request     import HTTPRedirectHandler, Request, build_opener

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


class Unfollowed(HTTPRedirectHandler):
    """
    The redirect handler that follows no redirect, so a request answered
    with one raises `HTTPError` carrying the redirect's own headers.
    """

    def redirect_request(self, *_: object):
        """
        Declines every redirect.
        """


def served(folder: Path, members: dict[str, bytes], *entries: TarInfo) -> Sample:
    """
    Packs `members` into a gzipped tar archive, each under its name, beside
    `entries`, each holding no data of its own, and writes it where the
    sample's address points once `folder` stands in for the Hub.

    Returns:
        The sample pinning the archive's digest, with `folder` as its Hub.
    """
    buffer = BytesIO()
    with TarFile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, data in members.items():
            info      = TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, BytesIO(data))

        for entry in entries:
            bundle.addfile(entry)

    sample = Sample(digest=sha256(buffer.getvalue()).hexdigest(), hub=folder.as_uri())
    (path := Path.from_uri(sample.address)).parent.mkdir(parents=True)
    path.write_bytes(buffer.getvalue())
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


def test_a_fetch_downloads_into_a_temporary_file_under_its_folder(
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that a fetch writes its download into a temporary file under the
    folder it extracts into.
    """
    sample = served(tmp_path / "hub", {"samples/wrf_sample.nc": b"CDF"})
    spy    = Mock(wraps=TemporaryFile)
    monkeypatch.setattr("thermur.environment.schemas.TemporaryFile", spy)

    sample.fetch(tmp_path / "clone")

    spy.assert_called_once_with(dir=tmp_path / "clone/data")


def test_a_fetch_extracts_an_archive_matching_its_digest(tmp_path: Path):
    """
    Asserts that a fetch downloads the archive, extracts every member under
    the folder, removes the downloaded archive, and returns the path of the
    member the graph reads.
    """
    sample = served(
        folder  = tmp_path / "hub",
        members = {"samples/README.md": b"#", "samples/wrf_sample.nc": b"CDF"}
    )

    fetched = sample.fetch(tmp_path / "clone")

    assert fetched == tmp_path / "clone/data/samples/wrf_sample.nc"
    assert fetched.read_bytes() == b"CDF"
    assert sorted(path.name for path in (tmp_path / "clone/data").rglob("*")) == [
        "README.md", "samples", "wrf_sample.nc"
    ]


def test_a_fetch_extracts_nothing_from_an_archive_with_another_digest(tmp_path: Path):
    """
    Asserts that a fetch whose archive hashes to another digest than the
    pinned one raises naming both, leaving the folder empty.
    """
    sample = served(tmp_path / "hub", {"samples/wrf_sample.nc": b"CDF"})

    with raises(ValueError, match=f"hashes to `{sample.digest}`"):
        sample.model_copy(update={"digest": "0" * 64}).fetch(tmp_path / "clone")

    assert list((tmp_path / "clone/data").iterdir()) == []


@mark.parametrize(
    ("kind", "target"),
    [
        param(SYMTYPE, "/etc/hosts", id="a-link-leading-out"),
        param(CHRTYPE, "", id="a-device")
    ]
)
def test_a_fetch_refuses_a_link_leading_out_and_a_device(
    kind     : bytes,
    target   : str,
    tmp_path : Path
):
    """
    Asserts that a fetch refuses an archive holding a link leading out of
    the folder it extracts into or a device, extracting neither.
    """
    entry = TarInfo("samples/member")
    entry.linkname, entry.type = target, kind

    with raises(FilterError):
        served(tmp_path / "hub", {}, entry).fetch(tmp_path / "clone")

    assert list((tmp_path / "clone/data").iterdir()) == []


def test_a_fetch_refuses_a_member_reaching_outside_its_folder(tmp_path: Path):
    """
    Asserts that a fetch refuses an archive whose member reaches outside the
    folder it extracts into, writing nothing outside it.
    """
    sample = served(tmp_path / "hub", {"../outside.txt": b"x"})

    with raises(FilterError):
        sample.fetch(tmp_path / "clone")

    assert not (tmp_path / "clone/outside.txt").exists()


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
    assert list((tmp_path / "clone/data").iterdir()) == []


def test_a_fetch_whose_download_fails_extracts_nothing(tmp_path: Path):
    """
    Asserts that a fetch whose download fails raises the error the download
    raised, leaving the folder empty.
    """
    with raises(URLError):
        Sample(hub=(tmp_path / "hub").as_uri()).fetch(tmp_path / "clone")

    assert list((tmp_path / "clone/data").iterdir()) == []


@mark.parametrize(
    ("path", "status"),
    [
        param(Sample().folder / Sample().member, 0, id="the-fetched-output"),
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
    Asserts that git ignores what a fetch extracts under the clone's `data/`
    folder, and no file under a `data` folder deeper in the clone.
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


def test_a_sample_addresses_its_archive_at_its_revision():
    """
    Asserts that the sample's address names the Hub, the dataset, the pinned
    revision, and the archive, in the form the Hub resolves a file of a
    dataset at a commit.
    """
    sample = Sample(hub="https://hub.test", repository="owner/data", revision="a" * 40)

    assert sample.address == (
        f"https://hub.test/datasets/owner/data/resolve/{'a' * 40}/samples.tar.gz"
    )


@mark.parametrize(
    ("speed_m_s", "held"),
    [
        param(8.0, nullcontext(), id="reaching-the-midline"),
        param(
            8.01,
            raises(ValidationError, match="reaches 1001 m .* domain 2000 m long"),
            id = "passing-the-midline"
        )
    ]
)
def test_a_threat_holds_its_front_inside_the_western_half(
    speed_m_s : float,
    held      : AbstractContextManager[object]
):
    """
    Asserts that a threat whose front and five times its `edge_m` ahead
    of it reach the midline of its domain by the last snapshot validates,
    and one passing it is refused, on a domain 2,000 m long that the front
    crosses for 100 s behind an edge of 40 m.
    """
    with held:
        ParametricThreat(
            grid = Grid(
                columns    = 50,
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


def test_a_sample_requests_its_archive_naming_the_package_alone():
    """
    Asserts that the request for the archive goes to the sample's address
    and names the package and its installed version as the client, the one
    header it sends.
    """
    request = Sample().request

    assert request.full_url == Sample().address
    assert request.header_items() == [("User-agent", f"thermur/{version('thermur')}")]


def test_a_second_fetch_writes_each_member_afresh(tmp_path: Path):
    """
    Asserts that fetching again over an earlier extraction writes each
    member afresh, so a member edited since reads as the archive holds it.
    """
    sample = served(tmp_path / "hub", {"samples/wrf_sample.nc": b"CDF"})
    sample.fetch(tmp_path / "clone").write_bytes(b"edited")

    assert sample.fetch(tmp_path / "clone").read_bytes() == b"CDF"


@mark.parametrize(
    "field",
    [
        param({"digest": "6f97"}, id="a-short-digest"),
        param(
            {"revision": "E7D4FC2CA1F99E82DFEEC3CA2F7BA1C714965913"},
            id = "an-uppercase-revision"
        )
    ]
)
def test_a_sample_refuses_a_pin_outside_its_form(field: dict[str, str]):
    """
    Asserts that a digest other than 64 lowercase hexadecimal digits, or a
    revision other than 40, is refused.
    """
    with raises(ValidationError):
        Sample(**field)


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
def test_the_hub_serves_the_pinned_archive_under_the_pinned_digest():
    """
    Asserts that the Hub answers the request for the archive with a redirect
    naming the pinned digest as the file's linked ETag and its linked size,
    read from the redirect's headers rather than from the archive.
    """
    request        = Sample().request
    request.method = "HEAD"

    with raises(HTTPError) as redirect:
        build_opener(Unfollowed).open(request)

    with redirect.value as response:
        assert response.headers["X-Linked-Etag"].strip('"') == Sample().digest
        assert int(response.headers["X-Linked-Size"]) == 490464998
