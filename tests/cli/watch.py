"""
Pins `thermur watch`, meaning the flight it flies for the Lookout and how it
lays it out, the files and the flight its server answers with, the page it
opens, and its exit where the clone holds no built site.
"""

from json    import loads
from numpy   import arange
from pathlib import Path
from pytest  import CaptureFixture, MonkeyPatch, fixture, raises
from socket  import socketpair
from xarray  import Dataset

from thermur.cli               import app, watch
from thermur.pipeline.graph    import Graph
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout


@fixture
def graph(monkeypatch: MonkeyPatch, tmp_path: Path) -> list[Graph]:
    """
    Stands in for computing the graph, recording each graph asked for its
    frames and answering with two frames of three seeds, whose values count
    up from zero, for the flock its settings size.

    Returns:
        Each graph computed, in order.
    """
    graphs: list[Graph] = []

    def compute(self: Graph, *steps: str) -> dict[str, Dataset]:
        """
        Records the graph and lays out the frames its plan sizes.
        """
        graphs.append(self)
        shape  = (3, 2, self.settings.plan.drones, 3)
        values = arange(0, 36 * shape[2], 2).reshape(shape) / 1000
        axes   = ("seed", "time", "drone", "axis")
        return {
            step: Dataset({"positions": (axes, values), "velocities": (axes, -values)})
            for step in steps
        }

    monkeypatch.setattr(Graph, "compute", compute)
    return graphs


def served(handler: type, request: bytes, **keywords: object) -> bytes:
    """
    Answers `request` through `handler` over a pair of joined sockets,
    without a server listening on any address.

    Returns:
        The bytes the handler wrote back, headers and body.
    """
    asking, answering = socketpair()
    with asking, answering:
        asking.sendall(request)
        asking.shutdown(1)
        handler(answering, ("watch", 0), None, **keywords)
        answering.close()
        return b"".join(iter(lambda: asking.recv(65536), b""))


def test_a_flight_is_flown_from_one_seed_and_laid_out_flat(graph: list[Graph]):
    """
    Asserts that the flight flies the size it is given from one seed under
    the graph's other settings, and lays out the first seed's frames flat,
    rounded to the centimeter, beside the size and the seconds between two
    frames.
    """
    start = Graph(settings=Settings())

    assert loads(watch.flown(8, start)) == {
        "drones"     : 8,
        "frame_s"    : start.settings.plan.frame_s,
        "positions"  : [round(k / 500, 2) for k in range(48)],
        "velocities" : [-round(k / 500, 2) for k in range(48)]
    }
    assert [done.settings.plan for done in graph] == [
        start.settings.plan.changed(drones=8, seeds=1)
    ]


def test_the_server_answers_the_flight_a_file_and_a_clean_address(
    graph    : list[Graph],
    tmp_path : Path
):
    """
    Asserts that the server answers `/flight` with the flight of the size
    its `drones` names as JSON, a file of the site with that file, and a
    page's address without its `.html` with that page.
    """
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods/lookout.html").write_text("<p>Lookout</p>")
    keywords = {"directory": str(tmp_path), "graph": Graph(settings=Settings())}

    def answer(path: str) -> tuple[bytes, bytes]:
        """
        Splits the server's answer to a request for `path`.
        """
        head, _, body = served(
            watch.Watching,
            f"GET {path} HTTP/1.0\r\n\r\n".encode(),
            **keywords
        ).partition(b"\r\n\r\n")
        return head, body

    flight, page, clean = answer("/flight?drones=9"), answer(
        "/methods/lookout.html"
    ), answer("/methods/lookout")

    assert b"application/json" in flight[0]
    assert loads(flight[1])["drones"] == 9
    assert [graph_.settings.plan.drones for graph_ in graph] == [9]
    assert page[0].startswith(b"HTTP/1.0 200") and page[1] == b"<p>Lookout</p>"
    assert clean[1] == page[1]


def test_watching_opens_the_lookout_on_the_served_flight(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that `thermur watch` serves the clone's built site on the
    loopback address at the port it is given, under the settings its flags
    set, and opens the Lookout reading its flights from `/flight`.
    """
    (tmp_path / "docs/.vitepress/dist").mkdir(parents=True)
    opened  : list[str] = []
    started : list[tuple[tuple[str, int], str, int]] = []

    class Server:
        """
        Stands in for the server, recording where it serves and what.
        """

        server_port = 8123

        def __enter__(self) -> "Server":
            return self

        def __exit__(self, *raised: object):
            pass

        def __init__(self, address: tuple[str, int], handler: object):
            started.append(
                (
                    address,
                    handler.keywords["directory"],
                    handler.keywords["graph"].settings.plan.drones
                )
            )

        def serve_forever(self):
            pass

    monkeypatch.setattr(watch, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(watch, "browse", opened.append)
    monkeypatch.setattr(
        watch,
        "Graph",
        lambda settings: Graph(checkout=Checkout(root=tmp_path), settings=settings)
    )

    app(
        ["watch", "--port", "8123", "--plan.drones", "40"],
        result_action = "return_value"
    )

    address = "http://127.0.0.1:8123/methods/lookout.html?flight=/flight"
    assert started == [(
        ("127.0.0.1", 8123),
        str(tmp_path.resolve() / "docs/.vitepress/dist"),
        40
    )]
    assert opened == [address]
    assert capsys.readouterr().out == f"Watching at {address}\n"


def test_watching_with_no_built_site_exits_naming_the_build(
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that `thermur watch` exits nonzero, serving nothing, where the
    clone holds no built site, naming the task that builds it.
    """
    monkeypatch.setattr(
        watch,
        "Graph",
        lambda settings: Graph(checkout=Checkout(root=tmp_path), settings=settings)
    )
    monkeypatch.setattr(watch, "ThreadingHTTPServer", None)

    with raises(SystemExit, match="mise run site:press"):
        app(["watch"], result_action="return_value")
