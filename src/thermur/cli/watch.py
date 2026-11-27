"""
Holds `watch`, the command `thermur watch` runs, serving the built site
from the clone and opening its Lookout on flights the package flies through
its graph.
"""

from cyclopts     import Parameter
from functools    import partial
from http.server  import SimpleHTTPRequestHandler, ThreadingHTTPServer
from json         import dumps
from numpy        import around, asarray
from pathlib      import Path
from typing       import Annotated
from urllib.parse import parse_qs, urlsplit
from webbrowser   import open as browse

from thermur.pipeline.graph    import Graph
from thermur.pipeline.settings import Settings


class Watching(SimpleHTTPRequestHandler):
    """
    Serves the built site, and at `/flight?drones=N` the flight of `N`
    drones `flown` flies through the graph.
    """

    def __init__(self, *arguments: object, graph: Graph, **keywords: object):
        self.graph = graph
        super().__init__(*arguments, **keywords)

    def do_GET(self):
        """
        Answers `/flight` with the flight its `drones` names, and every
        other path with the site's file there.
        """
        address = urlsplit(self.path)
        if address.path != "/flight":
            super().do_GET()
            return
        body = flown(int(parse_qs(address.query)["drones"][0]), self.graph)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path: str) -> str:
        """
        Finds the file at `path`, reading a page's address without its
        `.html` as the site's clean addresses write it.
        """
        found = Path(super().translate_path(path))
        page  = found.with_name(f"{found.name}.html")
        return str(page if not found.exists() and page.is_file() else found)


def flown(drones: int, graph: Graph) -> bytes:
    """
    Flies one flight of `drones` drones from the plan's seed through
    `graph`, laid out as the Lookout replays it: the flock's size, the
    seconds between two frames, and each frame's positions and velocities in
    turn, flat and rounded to the centimeter.

    Args:
        drones: The flock's size.
        graph: The graph whose settings the flight starts from.

    Returns:
        The flight as JSON.
    """
    settings = graph.settings
    plan     = settings.plan.changed(drones=drones, seeds=1)
    frames   = graph.model_copy(
        update = {"settings": settings.model_copy(update={"plan": plan})}
    ).compute("frames")["frames"]
    return dumps(
        {
            "drones"  : drones,
            "frame_s" : plan.frame_s,
            **{
                name: around(asarray(frames[name][0]), 2).ravel().tolist()
                for name in ("positions", "velocities")
            }
        }
    ).encode()


def watch(
    *,
    port     : int = 0,
    settings : Annotated[Settings | None, Parameter(name="*")] = None
):
    """
    Serves the site the clone built at `127.0.0.1`, opens its Lookout in
    the browser on flights the package flies through the graph, and serves
    until interrupted.

    Exits nonzero, serving nothing, where the clone holds no built site.

    Args:
        port: The port to serve on, any free one by default.
    """
    graph = Graph(settings=settings or Settings())
    built = graph.checkout.root / "docs/.vitepress/dist"
    if not built.is_dir():
        raise SystemExit(f"No built site at {built}: run `mise run site:press` first.")
    handler = partial(Watching, directory=str(built), graph=graph)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        site    = f"http://127.0.0.1:{server.server_port}"
        address = f"{site}/methods/lookout.html?flight=/flight"
        print(f"Watching at {address}")
        browse(address)
        server.serve_forever()
