"""
Holds the reader `thermur cut sample` opens a netCDF classic file through
over HTTP, meaning `Remote`, the file's address and the requests reading it,
`Header` and `Layout`, its header and the place of each variable in it, and
`RangedArray` and `RangedBackend`, through which xarray reads only the bytes
each selection covers.

The header follows the netCDF classic and 64-bit offset formats, per
https://docs.unidata.ucar.edu/nug/current/file_format_specifications.html,
and the backend follows xarray's guide to adding one, per
https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html.
"""

from collections.abc    import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses        import dataclass
from email              import message_from_bytes
from email.policy       import HTTP
from functools          import cached_property
from http.client        import HTTPConnection, HTTPException, HTTPSConnection
from importlib.metadata import version
from io                 import BufferedReader, RawIOBase
from math               import prod
from numpy              import (
    arange, array, diff, dtype, empty, flatnonzero, frombuffer, ndarray, split, unique
)
from pydantic        import BaseModel, InstanceOf
from queue           import Empty, SimpleQueue
from struct          import calcsize, unpack
from typing          import Any, BinaryIO, Self
from urllib.parse    import urlsplit
from urllib.request  import Request, urlopen
from xarray          import Dataset, Variable, decode_cf
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core     import indexing

from thermur.environment.schemas import Count, Positive


class Remote(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    A file served over HTTP, read in byte ranges packed into pieces, which a
    pool of threads requests over connections kept open between requests.
    """

    url: str
    """
    The address the file is read from, once every redirect is followed.
    """

    attempts: Count = 3
    """
    The most times one request is sent before its failure is raised.
    """

    connections: Count = 16
    """
    The most requests sent at once, each over a connection of its own.
    """

    piece_bytes: Count = 4 * 2**20
    """
    The most bytes one request covers.
    """

    range_chars: Count = 8000
    """
    The longest `Range` header one request carries, in characters, under the
    8190 bytes Apache's `LimitRequestFieldSize` allows a header by default.
    """

    timeout_s: Positive = 30.0
    """
    The longest a request waits on the connection or on one read, in seconds.
    """

    @property
    def header(self) -> Header:
        """
        Reads the file's header through a buffered reader over the ranges it
        covers, which fetches it in one request where it fits the buffer.
        """
        return Header.from_file(BufferedReader(RemoteFile(remote=self)))

    @property
    def headers(self) -> dict[str, str]:
        """
        Names the client in every request as the package at its installed
        version, such as `thermur/0.1.0`.
        """
        return {"User-Agent": f"thermur/{version('thermur')}"}

    @cached_property
    def idle(self) -> SimpleQueue[HTTPConnection]:
        """
        Holds each connection opened and not in use, which `http.client`
        opens again where the server closed it after a response.
        """
        return SimpleQueue()

    @cached_property
    def pool(self) -> ThreadPoolExecutor:
        """
        Starts the threads sending the requests, `connections` of them.
        """
        return ThreadPoolExecutor(self.connections)

    def close(self):
        """
        Stops the threads sending requests and closes every connection.
        """
        self.pool.shutdown()
        while not self.idle.empty():
            self.idle.get().close()

    def fetch(self, pieces: list[tuple[int, int]]) -> bytes:
        """
        Requests the ranges `pieces` names, each an offset and a length,
        through one connection, sending the request again on a failed
        connection, a stalled read, a body cut short, or a status other than
        200 or 206, and reading the parts of a multipart answer in whatever
        order the server sends them.

        Returns:
            The bytes of each range in the order `pieces` names them, cut
            short at the end of the file.

        Raises:
            HTTPException : Where every one of `attempts` fails, the last on
                            its answer, such as a status other than 200 or
                            206 or a body cut short.
            OSError       : Where every one of `attempts` fails, the last on
                            its connection or on a stalled read.
            ValueError    : Where the server answers with the whole file, or
                            sends no part holding one of the ranges.
        """
        address = urlsplit(self.url)
        try:
            connection = self.idle.get_nowait()
        except Empty:
            connection = (
                HTTPSConnection if address.scheme == "https" else HTTPConnection
            )(address.netloc, timeout=self.timeout_s)

        ranges = ",".join(
            f"{offset}-{offset + length - 1}" for offset, length in pieces
        )
        try:
            for _ in range(self.attempts):
                try:
                    connection.request(
                        "GET",
                        address._replace(netloc="", scheme="").geturl(),
                        headers = {"Range": f"bytes={ranges}"} | self.headers
                    )
                    with connection.getresponse() as response:
                        body = response.read()
                    if response.status == 200:
                        raise ValueError(
                            f"`{self.url}` answered a range request with status 200 "
                            "rather than 206"
                        )
                    if response.status != 206:
                        raise HTTPException(
                            f"`{self.url}` answered with status {response.status} "
                            f"{response.reason}"
                        )

                    break

                except (HTTPException, OSError) as error:
                    connection.close()
                    failure = error
            else:
                raise failure
        finally:
            self.idle.put(connection)

        parts = (
            [
                (part["Content-Range"], part.get_payload(decode=True))
                for part in message_from_bytes(
                    f"Content-Type: {response.headers['Content-Type']}\r\n\r\n".encode()
                    + body,
                    policy = HTTP
                ).iter_parts()
            ]
            if response.headers.get_content_type() == "multipart/byteranges"
            else [(response.headers["Content-Range"], body)]
        )
        starts = [
            (int(span.removeprefix("bytes ").split("-")[0]), data)
            for span, data in parts
        ]
        found = [
            next(
                (
                    data[offset - start : offset - start + length]
                    for start, data in starts
                    if start <= offset < start + len(data)
                ),
                None
            )
            for offset, length in pieces
        ]
        if None in found:
            raise ValueError(
                f"`{self.url}` sent no part holding the range from byte "
                f"{pieces[found.index(None)][0]}"
            )

        return b"".join(found)

    @classmethod
    def from_url(cls, url: str) -> Self:
        """
        Follows the redirects from `url` through one `HEAD` request, so
        every range is requested from the address serving the file.
        """
        remote = cls(url=url)
        with urlopen(
            Request(url, headers=remote.headers, method="HEAD"),
            timeout = remote.timeout_s
        ) as response:
            return remote.model_copy(update={"url": response.url})

    def packed(self, ranges: Iterable[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """
        Cuts `ranges`, each an offset and a length, into pieces of at most
        `piece_bytes`, and packs the pieces in order into requests holding
        at most `piece_bytes` between them and a `Range` header of at most
        `range_chars`.

        Returns:
            The pieces of each request, each an offset and a length.
        """
        requests: list[list[tuple[int, int]]] = []
        for offset, length in (
            (offset + start, min(self.piece_bytes, length - start))
            for offset, length in ranges
            for start in range(0, length, self.piece_bytes)
        ):
            chars = len(f"{offset}-{offset + length - 1},")
            if (
                not requests
                or size + length > self.piece_bytes
                or used + chars  > self.range_chars
            ):
                requests.append([])
                size = used = 0
            requests[-1].append((offset, length))
            size += length
            used += chars

        return requests

    def read(self, ranges: Iterable[tuple[int, int]]) -> bytes:
        """
        Reads `ranges`, each an offset and a length, as the requests
        `packed` makes of them, sent through `pool`.

        Returns:
            The bytes of each range, in the order `ranges` gives them.
        """
        return b"".join(self.pool.map(self.fetch, self.packed(ranges)))


class Layout(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One variable of a netCDF classic file as its header places it, its
    values written in row-major order, one record after another where it
    runs along the record dimension.
    """

    attributes: dict[str, Any]
    """
    The variable's attributes, each in the form netCDF4-python reads it.
    """

    begin: int
    """
    The offset of the variable's first value, in bytes from the start of the
    file.
    """

    dimensions: tuple[str, ...]
    """
    The names of the variable's dimensions, in order.
    """

    dtype: InstanceOf[dtype]
    """
    The big-endian type of each value as the file stores it.
    """

    record: bool
    """
    Whether the variable runs along the record dimension, its first.
    """

    shape: tuple[int, ...]
    """
    The size of each dimension, the record dimension's being the count of
    records written.
    """

    stride: int
    """
    The bytes from the start of one record of the variable to the start of
    the next, zero where it holds no record.
    """

    @property
    def native(self) -> dtype:
        """
        Reads the type of each value in the byte order of the machine.
        """
        return self.dtype.newbyteorder("=")

    def values(
        self,
        key  : tuple[int | slice, ...],
        *,
        read : Callable[[list[tuple[int, int]]], bytes]
    ) -> ndarray:
        """
        Reads the values `key` selects, one integer or slice for each
        dimension, through `read`, which takes one range for each run of
        adjacent values selected in each record, as an offset and a length,
        and returns their bytes one after another.

        Returns:
            The values selected, in the byte order of the machine.
        """
        records, inner = (key[0], key[1:]) if self.record else (None, key)
        block          = self.shape[self.record :]
        picked         = arange(prod(block)).reshape(block)[inner]
        starts         = arange(self.shape[0])[records] if self.record else array(0)
        if not (picked.size and starts.size):
            return empty(starts.shape + picked.shape, self.native)

        (kept, inverse), size = unique(picked, return_inverse=True), self.dtype.itemsize
        runs = split(kept, flatnonzero(diff(kept) != 1) + 1)
        data = read(
            [
                (self.begin + start * self.stride + int(run[0]) * size, run.size * size)
                for start in starts.ravel().tolist()
                for run in runs
            ]
        )
        return frombuffer(data, self.dtype).reshape(starts.size, kept.size)[
            :, inverse
        ].reshape(starts.shape + picked.shape).astype(self.native)


class Header(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The header of a netCDF classic or 64-bit offset file, holding its global
    attributes, its dimensions, and the layout of each variable.
    """

    attributes: dict[str, Any]
    """
    The file's global attributes, each in the form netCDF4-python reads it.
    """

    dimensions: dict[str, int]
    """
    The size of each dimension by its name, the record dimension's being the
    count of records written.
    """

    variables: dict[str, Layout]
    """
    The layout of each variable by its name, in the order the file lists
    them.
    """

    @classmethod
    def from_file(cls, file: BinaryIO) -> Self:
        """
        Parses the header at the start of `file`, the record dimension being
        the one the header gives a size of zero.

        Each record holds every record variable's values for that record,
        each padded to four bytes unless only one variable runs along the
        record dimension.

        Raises:
            ValueError: Where `file` opens on neither the classic nor the
                        64-bit offset format's magic number, naming what it
                        opens on.
        """
        magic, count = unpack(">4sI", file.read(8))
        if magic not in (b"CDF\x01", b"CDF\x02"):
            raise ValueError(
                f"The file opens on {magic!r} rather than the magic number of the "
                "netCDF classic or 64-bit offset format"
            )

        offset = ">Q" if magic == b"CDF\x02" else ">I"
        sizes  = listed(file, lambda: (string(file), integer(file)))
        header = cls(
            attributes = dict(listed(file, lambda: attribute(file))),
            dimensions = {key: size or count for key, size in sizes},
            variables  = {
                key: Layout(
                    attributes = attrs,
                    begin      = begin,
                    dimensions = tuple(dimension for dimension, _ in dimensions),
                    dtype      = kind,
                    record     = bool(dimensions) and not dimensions[0][1],
                    shape      = tuple(size or count for _, size in dimensions),
                    stride     = 0
                )
                for key, dimensions, attrs, kind, _, begin in listed(
                    file = file,
                    item = lambda: (
                        string(file),
                        [sizes[integer(file)] for _ in range(integer(file))],
                        dict(listed(file, lambda: attribute(file))),
                        typed(integer(file)),
                        integer(file),
                        unpack(offset, file.read(calcsize(offset)))[0]
                    )
                )
            }
        )
        records = [
            prod(layout.shape[1:]) * layout.dtype.itemsize
            for layout in header.variables.values()
            if layout.record
        ]
        stride = (
            records[0]
            if len(records) == 1
            else sum(-(-size // 4) * 4 for size in records)
        )
        return header.model_copy(
            update = {
                "variables": {
                    key: layout.model_copy(update={"stride": stride * layout.record})
                    for key, layout in header.variables.items()
                }
            }
        )


@dataclass(frozen=True)
class RangedArray(BackendArray):
    """
    One variable of a `Remote` file as xarray reads it lazily, each
    selection read as `Layout.values` reads it.
    """

    layout: Layout
    """
    The variable's place in the file.
    """

    remote: Remote
    """
    The file the variable is read from.
    """

    def __getitem__(self, key: indexing.ExplicitIndexer) -> ndarray:
        """
        Reads the values `key` selects, reading the integers and slices of
        an outer or vectorized key from the file and applying the rest in
        memory, through `explicit_indexing_adapter`.
        """
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            lambda basic: self.layout.values(basic, read=self.remote.read)
        )

    @property
    def dtype(self) -> dtype:
        """
        Reads the type of each value in the byte order of the machine.
        """
        return self.layout.native

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Reads the size of each dimension.
        """
        return self.layout.shape


class RangedBackend(BackendEntrypoint):
    """
    The xarray backend opening a netCDF classic file served over HTTP,
    passed to `open_dataset` as its `engine`, which decodes the file as it
    decodes one the `netcdf4` engine opens.
    """

    description = "Opens a netCDF classic file over HTTP, reading byte ranges"

    def open_dataset(
        self,
        filename_or_obj : str,
        *,
        drop_variables  : str | Iterable[str] | None = None
    ) -> Dataset:
        """
        Opens the file at the address `filename_or_obj` names, reading its
        header and leaving every value unread until a caller reads it.
        """
        remote  = Remote.from_url(filename_or_obj)
        header  = remote.header
        dataset = decode_cf(
            Dataset(
                {
                    key: Variable(
                        layout.dimensions,
                        indexing.LazilyIndexedArray(
                            RangedArray(layout=layout, remote=remote)
                        ),
                        layout.attributes
                    )
                    for key, layout in header.variables.items()
                },
                attrs = header.attributes
            ),
            drop_variables = drop_variables
        )
        dataset.set_close(remote.close)
        return dataset


@dataclass
class RemoteFile(RawIOBase):
    """
    A `Remote` file read from its start as a stream, each read requesting
    the range it fills.
    """

    remote: Remote
    """
    The file read.
    """

    position: int = 0
    """
    The offset the next read starts at.
    """

    def readable(self) -> bool:
        """
        Reports that the stream reads.
        """
        return True

    def readinto(self, buffer: memoryview) -> int:
        """
        Fills `buffer` from `position`, or as much of it as the file holds.

        Returns:
            The count of bytes read.
        """
        data                = self.remote.fetch([(self.position, len(buffer))])
        buffer[: len(data)] = data
        self.position      += len(data)
        return len(data)


def attribute(file: BinaryIO) -> tuple[str, Any]:
    """
    Reads one attribute from `file`, its text as a string with every null
    byte dropped and its numbers as an array in the byte order of the
    machine, or as a scalar where it holds one, as netCDF4-python reads it.
    """
    key, kind, count = string(file), typed(integer(file)), integer(file)
    raw = file.read(-(-count * kind.itemsize // 4) * 4)[: count * kind.itemsize]
    if kind.char == "S":
        return key, raw.decode(errors="replace").replace("\x00", "")

    values = frombuffer(raw, kind).astype(kind.newbyteorder("="))
    return key, values[0] if count == 1 else values


def integer(file: BinaryIO) -> int:
    """
    Reads one unsigned 32-bit big-endian integer from `file`.
    """
    return unpack(">I", file.read(4))[0]


def listed[T](file: BinaryIO, item: Callable[[], T]) -> list[T]:
    """
    Reads one list of the header from `file`, its tag and its count followed
    by each item `item` reads, the count being zero for an absent list.
    """
    integer(file)
    return [item() for _ in range(integer(file))]


def string(file: BinaryIO) -> str:
    """
    Reads one name from `file`, its length followed by its UTF-8 bytes
    padded to four.
    """
    length = integer(file)
    return file.read(-(-length // 4) * 4)[:length].decode()


def typed(code: int) -> dtype:
    """
    Reads the big-endian type the netCDF type `code` names, from `NC_BYTE`
    at 1 through `NC_DOUBLE` at 6.
    """
    return dtype((">i1", "S1", ">i2", ">i4", ">f4", ">f8")[code - 1])
