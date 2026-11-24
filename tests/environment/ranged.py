"""
Pins the reader `thermur cut sample` opens a netCDF classic file through
over HTTP, served here from the loopback address by `Served`:

- The header, read as netCDF4-python reads it, in the classic and the
  64-bit offset formats, beside the refusal of any other
- The values of each variable, read as the `netcdf4` engine reads them,
  through the redirect the address answers with
- The bytes a selection requests, one range per run of adjacent values in
  each record it covers, packed into multipart requests
- The pieces the ranges are cut into and packed as, and the connections
  kept open between requests
- The retry of a failed request, and the refusal of a server that ignores
  the range or leaves a range out of its multipart answer
- The record `Served` keeps of each range, written before the client can
  read the range
- The traceback `Served` prints for a failure in its handler, beside the
  silence of a connection the client closed or reset
"""

from collections.abc import Iterable, Iterator
from common.served   import Served
from contextlib      import closing, suppress
from http.client     import HTTPException, RemoteDisconnected
from io              import BytesIO
from math            import prod
from numpy           import arange, full, int16
from pathlib         import Path
from pytest          import CaptureFixture, FixtureRequest, fixture, mark, param, raises
from socket          import SOL_SOCKET, SO_LINGER
from struct          import pack
from time            import sleep
from xarray          import Dataset, open_dataset
from xarray.testing  import assert_identical

from thermur.environment.ranged  import Header, RangedBackend, Remote
from thermur.environment.schemas import Grid, ParametricThreat

pytestmark = [mark.enable_socket, mark.allow_hosts(["127.0.0.1"])]


class Lagging[T](list[T]):
    """
    The entries a server records, pausing for 0.3 s before each record, as a
    loaded runner holds back the thread serving a request.
    """

    def append(self, entry: T):
        """
        Records `entry` after the pause.
        """
        sleep(0.3)
        super().append(entry)

    def extend(self, entries: Iterable[T]):
        """
        Records `entries` after the pause.
        """
        sleep(0.3)
        super().extend(entries)


@fixture
def remote(served: Served) -> Iterator[Remote]:
    """
    Opens `served` as a `Remote` for the length of the case, through the
    redirect its address answers with.
    """
    with served.serving() as address, closing(Remote.from_url(address)) as remote:
        yield remote


@fixture
def served(threat: Path) -> Served:
    """
    Serves `threat` from the loopback address.
    """
    return Served(path=threat)


@fixture
def threat(tmp_path: Path) -> Path:
    """
    Writes a still threat on a grid of 20 columns, 3 rows, and 5 levels
    over 3 snapshots as a netCDF file in the 64-bit offset format, its
    `Time` unlimited, under the global attributes WRF writes as text, as one
    number, and as several.
    """
    path = tmp_path / "wrfout"
    ParametricThreat(
        grid = Grid(
            columns   = 20,
            levels    = 5,
            rows      = 3,
            snapshots = 3
        ),
        speed_m_s = 0.0
    ).dataset.assign_attrs(
        TITLE    = " OUTPUT FROM WRF V3.4 MODEL",
        TRUELATS = arange(2, dtype="float32")
    ).to_netcdf(
        path,
        engine         = "netcdf4",
        format         = "NETCDF3_64BIT",
        unlimited_dims = ["Time"]
    )
    return path


@fixture(
    params = [
        param("NETCDF3_CLASSIC", id="classic"),
        param("NETCDF3_64BIT", id="offset"),
        param("lone", id="one-record-variable")
    ]
)
def written(request: FixtureRequest, tmp_path: Path) -> Path:
    """
    Writes a netCDF file in the classic or the 64-bit offset format, holding
    a variable along the record dimension and one without it, beside a text
    variable of 19 characters a record, as WRF writes `Times`, which the
    format pads to 20 between the record's variables, or a file whose one
    record variable holds three 16-bit integers a record, which the format
    writes with no padding between records.
    """
    path = tmp_path / "written.nc"
    lone = request.param == "lone"
    Dataset(
        {
            "count"  : (
                ("Time", "x"),
                arange(9, dtype=int16).reshape(3, 3),
                {"units": "1"}
            ),
            "ground" : (("y", "x"), arange(6.0).reshape(2, 3), {"scale": 0.5}),
            "Times"  : (("Time", "DateStrLen"), full((3, 19), b"0", "S1"))
        }
    ).drop_vars(["ground", "Times"] if lone else []).to_netcdf(
        path,
        engine         = "netcdf4",
        format         = "NETCDF3_CLASSIC" if lone else request.param,
        unlimited_dims = ["Time"]
    )
    return path


def test_a_connection_the_client_resets_ends_with_nothing_printed(
    capsys : CaptureFixture[str],
    served : Served
):
    """
    Asserts that a client resetting its connection once it has read an
    answer, closing it with a linger of no seconds rather than closing it in
    order, ends the connection with nothing printed.
    """
    with served.serving() as address:
        remote = Remote.from_url(address)

        assert remote.fetch([(0, 4)]) == b"CDF\x02"

        connection = remote.idle.get()
        connection.sock.setsockopt(SOL_SOCKET, SO_LINGER, pack("ii", 1, 0))
        connection.close()

    assert capsys.readouterr().err == ""


def test_a_crop_across_columns_requests_each_row_in_one_multipart_request(
    served : Served,
    threat : Path
):
    """
    Asserts that reading three of the columns of `T` in one record requests
    one range for each row of each level, all in one request, and reads the
    values the `netcdf4` engine reads from the parts the server sends in the
    reverse order.
    """
    with threat.open("rb") as file:
        layout = Header.from_file(file).variables["T"]

    *_, columns = layout.shape
    size        = layout.dtype.itemsize
    with (
        served.serving() as address,
        open_dataset(address, engine=RangedBackend) as ranged,
        open_dataset(threat, engine="netcdf4") as local
    ):
        served.requests.clear()
        served.sent.clear()
        selection = {"Time": 1, "west_east": slice(2, 5)}

        assert_identical(
            ranged.T.isel(selection).load(),
            local.T.isel(selection).load()
        )

    assert (len(served.requests), sorted(served.sent)) == (
        1,
        [
            (layout.begin + layout.stride + (row * columns + 2) * size, 3 * size)
            for row in range(prod(layout.shape[1:-1]))
        ]
    )


@mark.parametrize(
    "fault",
    [param("cut", id="cut-short"), param("unavailable", id="503")]
)
def test_a_failed_range_is_requested_again(fault: str, remote: Remote, served: Served):
    """
    Asserts that a request whose body is cut short, or whose server answers
    with a status of 503, is sent again and read whole.
    """
    served.faults[:] = [fault]

    assert remote.fetch([(0, 4)]) == b"CDF\x02"


def test_a_failure_in_the_handler_prints_its_traceback(
    capsys   : CaptureFixture[str],
    tmp_path : Path
):
    """
    Asserts that a request the handler of `Served` fails on, here for a file
    gone from the disk, closes the connection with no answer and prints the
    traceback of the failure.
    """
    with (
        Served(path=tmp_path / "missing").serving() as address,
        raises(RemoteDisconnected)
    ):
        Remote.from_url(address)

    assert "FileNotFoundError" in capsys.readouterr().err


def test_a_failure_on_every_attempt_is_raised(remote: Remote, served: Served):
    """
    Asserts that a range failing on each of `Remote.attempts` raises the
    last failure.
    """
    served.faults[:] = ["unavailable"] * remote.attempts

    with raises(HTTPException, match="503"):
        remote.fetch([(0, 4)])


@mark.parametrize(
    ("selection", "records", "levels"),
    [
        param(
            {"Time": slice(1, 3), "bottom_top": slice(0, 2)},
            [1, 2],
            range(0, 2),
            id = "records"
        ),
        param(
            {"Time": 1, "bottom_top": slice(1, 4)},
            [1],
            range(1, 4),
            id = "one-record"
        ),
        param(
            {"Time": slice(0, 0), "bottom_top": slice(0, 2)},
            [],
            range(0, 2),
            id = "no-record"
        )
    ]
)
def test_a_selection_requests_one_range_per_record_it_covers(
    levels    : range,
    records   : list[int],
    selection : dict[str, int | slice],
    served    : Served,
    threat    : Path
):
    """
    Asserts that reading the levels `selection` keeps of `T` requests one
    range per record, running from the first level kept to the last, and
    nothing where it keeps no record.
    """
    with threat.open("rb") as file:
        layout = Header.from_file(file).variables["T"]

    plane = prod(layout.shape[2:]) * layout.dtype.itemsize
    with (
        served.serving() as address,
        open_dataset(address, engine=RangedBackend) as dataset
    ):
        served.sent.clear()
        dataset.T.isel(selection).load()

    assert sorted(served.sent) == [
        (
            layout.begin + record * layout.stride + levels.start * plane,
            len(levels) * plane
        )
        for record in records
    ]


def test_a_file_in_another_format_is_refused_naming_its_magic_number():
    """
    Asserts that a file opening on another magic number than the classic or
    the 64-bit offset format's, such as HDF5's, is refused naming it.
    """
    with raises(ValueError, match=r"opens on b'\\x89HDF'"):
        Header.from_file(BytesIO(b"\x89HDF\r\n\x1a\n"))


def test_a_file_served_over_http_reads_as_the_netcdf4_engine_reads_it(written: Path):
    """
    Asserts that a file opened through `RangedBackend` reads, attributes
    included, as the same file opened through the `netcdf4` engine.
    """
    with (
        Served(path=written).serving() as address,
        open_dataset(address, engine=RangedBackend) as ranged,
        open_dataset(written, engine="netcdf4") as local
    ):
        assert_identical(ranged.load(), local.load())


def test_a_multipart_answer_leaving_out_a_range_is_refused(
    remote : Remote,
    served : Served
):
    """
    Asserts that a multipart answer holding no part for one of the ranges
    requested is refused, naming the byte that range starts at.
    """
    served.faults[:] = ["dropped"]

    with raises(ValueError, match="no part holding the range from byte 0$"):
        remote.fetch([(0, 4), (8, 4)])


def test_a_part_holding_several_ranges_is_read_for_each(
    remote : Remote,
    served : Served,
    threat : Path
):
    """
    Asserts that a multipart answer whose one part runs from the first range
    requested to the last is read for each range at its own offset in it.
    """
    served.faults[:] = ["merged"]
    data             = threat.read_bytes()

    assert remote.fetch([(0, 4), (8, 4)]) == data[:4] + data[8:12]


def test_a_range_is_recorded_before_its_answer_is_read(remote: Remote, served: Served):
    """
    Asserts that `Served.requests` and `Served.sent` hold a range once the
    client has read its answer, even where the server pauses before each
    record.
    """
    served.requests = Lagging()
    served.sent     = Lagging()

    assert remote.fetch([(0, 4)]) == b"CDF\x02"
    assert (served.requests, served.sent) == (["bytes=0-3"], [(0, 4)])


def test_a_server_ignoring_the_range_is_refused(
    capsys : CaptureFixture[str],
    served : Served
):
    """
    Asserts that a server answering a range request with the whole file and
    a status of 200 is refused, naming the status, and prints nothing once
    the client resets the connection it refused the answer on.
    """
    served.faults[:] = ["whole"]
    with served.serving() as address, closing(Remote.from_url(address)) as remote:
        with raises(ValueError, match="status 200 rather than 206"):
            remote.fetch([(0, 4)])

    assert capsys.readouterr().err == ""


def test_a_stalled_read_is_requested_again_over_a_new_connection(threat: Path):
    """
    Asserts that a request whose answer stalls past `Remote.timeout_s` is
    sent again over its connection opened anew, and read whole.
    """
    with Served(faults=["stalled"], path=threat).serving() as address:
        remote = Remote.from_url(address).model_copy(update={"timeout_s": 0.2})
        with closing(remote):
            assert remote.fetch([(0, 4)]) == b"CDF\x02"


def test_an_address_is_read_where_its_redirect_leads(threat: Path):
    """
    Asserts that the address a file is opened at is followed through its
    redirect, so every range is requested from the address serving it.
    """
    with Served(path=threat).serving() as address:
        assert Remote.from_url(address).url == address.replace("/moved/", "/files/")


@mark.parametrize(
    ("fault", "pieces"),
    [
        param("dropped", [(0, 4), (8, 4)], id="part-left-out"),
        param("merged", [(0, 4), (8, 4)], id="parts-merged"),
        param("whole", [(0, 4)], id="whole-file")
    ]
)
def test_an_answer_with_a_fault_is_left_unrecorded(
    fault  : str,
    pieces : list[tuple[int, int]],
    remote : Remote,
    served : Served
):
    """
    Asserts that an answer the server sends with a fault adds neither its
    `Range` header nor its ranges to the record, whether the client reads
    the answer or refuses it.
    """
    served.requests.clear()
    served.sent.clear()
    served.faults[:] = [fault]

    with suppress(ValueError):
        remote.fetch(pieces)

    assert (served.requests, served.sent) == ([], [])


@mark.parametrize(
    ("kept_alive", "opened"),
    [param(3, 3, id="closed-every-third"), param(100, 1, id="kept-open")]
)
def test_a_connection_is_kept_open_until_the_server_closes_it(
    kept_alive : int,
    opened     : int,
    threat     : Path
):
    """
    Asserts that nine requests sent one after another over one connection
    open it once where the server answers each of them over it, and open it
    again each time the server closes it, here after every third.
    """
    served = Served(kept_alive=kept_alive, path=threat)
    with served.serving() as address:
        remote = Remote.from_url(address).model_copy(
            update = {"connections": 1, "piece_bytes": 16}
        )
        served.opened = 0
        with closing(remote):
            assert remote.read([(0, 144)]) == threat.read_bytes()[:144]

    assert (len(served.requests), served.opened) == (9, opened)


@mark.parametrize(
    ("ranges", "requests"),
    [
        param([(0, 40)], [[(0, 16)], [(16, 16)], [(32, 8)]], id="cut"),
        param(
            [(0, 8), (100, 8), (200, 8)],
            [[(0, 8), (100, 8)], [(200, 8)]],
            id = "bytes"
        ),
        param(
            [(0, 1), (10, 1), (20, 1), (30, 1)],
            [[(0, 1), (10, 1)], [(20, 1), (30, 1)]],
            id = "header"
        ),
        param([], [], id="none")
    ]
)
def test_ranges_are_cut_and_packed_into_requests_in_order(
    ranges   : list[tuple[int, int]],
    requests : list[list[tuple[int, int]]]
):
    """
    Asserts that a range longer than `piece_bytes` is cut into pieces, and
    that pieces pack in order into one request until another would pass
    `piece_bytes` between them or a `Range` header of `range_chars`.
    """
    remote = Remote(piece_bytes=16, range_chars=12, url="http://data.invalid/file")

    assert remote.packed(ranges) == requests
