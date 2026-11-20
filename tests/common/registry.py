"""
Holds `Registry`, the stand-in a lock task test resolves the package `aged`
against on the loopback address, answering as a PyPI simple index and as the
npm registry with the date each release was published, and `serving`, which
runs it.
"""

from base64          import b64encode
from collections.abc import Iterator
from contextlib      import contextmanager
from datetime        import datetime
from hashlib         import sha256, sha512
from http.server     import BaseHTTPRequestHandler, ThreadingHTTPServer
from io              import BytesIO
from json            import dumps
from threading       import Thread
from zipfile         import ZipFile


@contextmanager
def serving(published: dict[str, datetime]) -> Iterator[str]:
    """
    Serves the releases `published` dates from a thread for the length of
    the block.

    Yields:
        The server's address under the loopback address and the port the
        system chose, with no closing slash.
    """
    with ThreadingHTTPServer(("127.0.0.1", 0), Registry) as server:
        server.published = published
        thread           = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


class Registry(BaseHTTPRequestHandler):
    """
    Answers for the package `aged` at each release the server's `published`
    map dates, as the JSON simple index uv reads at `/simple/aged/`, each
    wheel under `/files/`, and the document Bun reads at `/aged`, whose
    integrity hashes the version alone, since `bun install --lockfile-only`
    downloads no tarball.
    """

    @property
    def address(self) -> str:
        """
        Reads the address the request reached the server at, with its scheme
        and no closing slash.
        """
        return f"http://{self.headers['Host']}"

    @property
    def published(self) -> dict[str, str]:
        """
        Writes the date each release was published in the form both
        registries publish it, by release.
        """
        return {
            version: f"{date:%Y-%m-%dT%H:%M:%S.000Z}"
            for version, date in self.server.published.items()
        }

    def do_GET(self):
        """
        Writes the index, the wheel, or the npm document the path names,
        answering any other path with `404`.
        """
        match self.path:
            case "/simple/aged/":
                kind = "application/vnd.pypi.simple.v1+json"
                body = self.index()
            case "/aged":
                kind = "application/json"
                body = self.document()
            case path if path.startswith("/files/aged-"):
                kind = "application/octet-stream"
                body = wheel(path.split("-")[1])
            case _:
                self.send_error(404)
                return

        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        """
        Refuses every `HEAD` with `405`, so uv downloads a wheel whole
        rather than reading its metadata through ranges.
        """
        self.send_error(405)

    def document(self) -> bytes:
        """
        Writes the npm document Bun reads, listing each release beside the
        date it was published and naming the newest under `latest`.
        """
        return dumps(
            {
                "dist-tags" : {"latest": max(self.published)},
                "name"      : "aged",
                "time"      : self.published,
                "versions"  : {
                    version: {
                        "dist"    : {
                            "integrity" : "sha512-"
                            + b64encode(sha512(version.encode()).digest()).decode(),
                            "tarball"   : f"{self.address}/aged/-/aged-{version}.tgz"
                        },
                        "name"    : "aged",
                        "version" : version
                    }
                    for version in self.published
                }
            }
        ).encode()

    def index(self) -> bytes:
        """
        Writes the JSON simple index uv reads, listing each release's wheel
        beside its hash and the date it was uploaded.
        """
        return dumps(
            {
                "files" : [
                    {
                        "filename"    : f"aged-{version}-py3-none-any.whl",
                        "hashes"      : {"sha256": sha256(wheel(version)).hexdigest()},
                        "upload-time" : date,
                        "url": f"{self.address}/files/aged-{version}-py3-none-any.whl"
                    }
                    for version, date in self.published.items()
                ],
                "name"  : "aged",
                "meta"  : {"api-version": "1.1"}
            }
        ).encode()

    def log_message(self, *_: object):
        """
        Writes nothing for each request, which the server would otherwise
        log to standard error.
        """


def wheel(version: str) -> bytes:
    """
    Writes the pure-Python wheel of the release `version` of `aged`, an
    empty module beside the metadata uv reads its name and version from.
    """
    folder  = f"aged-{version}.dist-info"
    archive = BytesIO()
    with ZipFile(archive, "w") as written:
        written.writestr("aged/__init__.py", "")
        written.writestr(
            f"{folder}/METADATA",
            f"Metadata-Version: 2.1\nName: aged\nVersion: {version}\n"
        )
        written.writestr(
            f"{folder}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: thermur\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        written.writestr(f"{folder}/RECORD", "")

    return archive.getvalue()
