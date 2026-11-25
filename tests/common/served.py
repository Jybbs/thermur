"""
Holds `Served`, one file a test serves over HTTP on the loopback address,
answering range requests as the Apache server the WRF-SFIRE dataset
publishes through does, behind a redirect from another path, failing
the requests a case lines up, and printing the traceback of a failure
in a handler before the case ends unless the client closed or reset the
connection.
"""

from collections.abc import Iterator
from contextlib      import contextmanager, suppress
from http.server     import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib         import Path
from pydantic        import BaseModel
from threading       import Thread
from time            import sleep
from typing          import Literal


class Served(BaseModel, extra="forbid", use_attribute_docstrings=True):
    """
    One file served at `/files/<name>` and redirected to from
    `/moved/<name>`, beside each range the server sent and the failures it
    has yet to answer with.
    """

    path: Path
    """
    The file served.
    """

    faults: list[
        Literal["cut", "dropped", "merged", "stalled", "unavailable", "whole"]
    ] = []
    """
    The answers the next requests take, one each in order, meaning a body
    cut short, a multipart body leaving out the part of the first range, a
    multipart body of one part running from the first range to the last, a
    body whose first byte is followed by a pause of a second before the
    connection closes, a status of 503, and the whole file with a status of
    200.
    """

    kept_alive: int = 100
    """
    The most requests one connection answers before the server closes it,
    100 by default, as Apache's `MaxKeepAliveRequests` sets it.
    """

    opened: int = 0
    """
    The count of connections the server accepted.
    """

    requests: list[str] = []
    """
    The `Range` header of each request answered with no fault, recorded
    before the body is written.
    """

    sent: list[tuple[int, int]] = []
    """
    The offset and the length of each range sent whole, in the order sent,
    recorded before the body is written.
    """

    @property
    def bytes_sent(self) -> int:
        """
        Sums the lengths of every range sent whole.
        """
        return sum(length for _, length in self.sent)

    @contextmanager
    def serving(self) -> Iterator[str]:
        """
        Serves the file from a thread for the length of the block, and waits
        on leaving it for each connection's thread to end.

        Yields:
            The address the file is redirected to from, under the loopback
            address and the port the system chose.
        """
        served = self

        class Handler(BaseHTTPRequestHandler):
            """
            Answers each request for the file, its range, or its redirect,
            keeping the connection open between requests as HTTP/1.1 does.
            """

            protocol_version = "HTTP/1.1"

            def do_GET(self):
                """
                Answers with the range the request names, or with a
                multipart body holding each range it names in the reverse
                order, each part's `Content-Range` in the form the dataset's
                server writes it, with no unit and no total, or with the
                fault lined up next.
                """
                if self.moved():
                    return

                data   = served.path.read_bytes()
                header = self.headers["Range"]
                spans  = [
                    (int(start), min(int(end), len(data) - 1))
                    for start, end in (
                        span.split("-") for span in (
                            header.removeprefix("bytes=")
                                  .split(",")
                        )
                    )
                ]
                fault          = served.faults.pop(0) if served.faults else None
                self.answered += 1
                if fault == "unavailable":
                    self.send_error(503)
                    return

                (first, last), *rest = spans
                body, kind           = (
                    (
                        b"".join(
                            b"\r\n--parts\r\nContent-Range: %d-%d/*\r\n\r\n" % span
                            + data[span[0] : span[1] + 1]
                            for span in (
                                [(first, spans[-1][1])]
                                if fault == "merged"
                                else reversed(spans[fault == "dropped" :])
                            )
                        ) + b"\r\n--parts--\r\n",
                        "multipart/byteranges; boundary=parts"
                    )
                    if rest
                    else (
                        data if fault == "whole" else data[first : last + 1],
                        "application/octet-stream"
                    )
                )

                self.send_response(200 if fault == "whole" else 206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", kind)
                if not rest:
                    self.send_header(
                        "Content-Range",
                        f"bytes {first}-{last}/{len(data)}"
                    )
                if fault == "cut" or self.answered == served.kept_alive:
                    self.close_connection = True
                    self.send_header("Connection", "close")
                self.end_headers()
                if fault == "cut":
                    self.wfile.write(body[: len(body) // 2])
                    return

                if fault == "stalled":
                    self.wfile.write(body[:1])
                    self.wfile.flush()
                    sleep(1.0)
                    self.close_connection = True
                    return

                if fault is None:
                    served.requests.append(header)
                    served.sent.extend((start, end - start + 1) for start, end in spans)
                self.wfile.write(body)

            def do_HEAD(self):
                """
                Answers with the size of the file, or with its redirect.
                """
                if self.moved():
                    return

                self.send_response(200)
                self.send_header("Content-Length", str(served.path.stat().st_size))
                self.end_headers()

            def handle(self):
                """
                Answers each request over the connection until either end
                closes it, ending with nothing printed where the client
                resets the connection or closes it while the server writes.
                """
                with suppress(ConnectionError):
                    super().handle()

            def log_message(self, *_: object):
                """
                Writes nothing, leaving the output of a case to the case.
                """

            def moved(self) -> bool:
                """
                Redirects a request for `/moved/<name>` to `/files/<name>`.

                Returns:
                    Whether the request was redirected.
                """
                if not self.path.startswith("/moved/"):
                    return False

                self.send_response(302)
                self.send_header("Location", self.path.replace("/moved/", "/files/"))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True

            def setup(self):
                """
                Counts the connection the server accepted, and the requests
                answered over it from none, before the handler reads from
                it.
                """
                self.answered  = 0
                served.opened += 1
                super().setup()

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            server.daemon_threads = False
            thread                = Thread(target=server.serve_forever)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}/moved/{self.path.name}"
            finally:
                server.shutdown()
                thread.join()
