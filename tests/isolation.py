"""
Pins the isolation the suite keeps from the machine running it, covering
the shell variables each test starts without, the home folder it reads,
the block that stops a test without the `network` mark from opening a
connection, the Unix socket a process pool reaches its server over, and the
mark that lets a `network` test open a connection.
"""

from collections.abc    import Iterator
from common.environment import CLEARED
from os                 import environ
from pathlib            import Path
from pytest             import FixtureRequest, MonkeyPatch, fixture, mark, raises, warns
from pytest_socket      import SocketBlockedError
from socket             import create_connection, socketpair
from tests.conftest     import pytest_collection_modifyitems


@fixture(params=CLEARED, scope="module")
def name(request: FixtureRequest) -> Iterator[str]:
    """
    Sets the variable `request.param` names and yields that name.

    A module-scoped fixture runs before the function-scoped `environment`
    fixture, so the variable is set on every machine by the time
    `environment` clears it, a CI runner that never sets it included.
    """
    with MonkeyPatch.context() as patched:
        patched.setenv(request.param, "1")
        yield request.param


def test_a_network_test_gets_the_mark_that_opens_the_socket(request: FixtureRequest):
    """
    Asserts that the collection hook in `tests/conftest.py` gives a test
    carrying the `network` mark pytest-socket's `enable_socket` mark, which
    lets that test open a connection.
    """
    request.node.add_marker(mark.network)
    pytest_collection_modifyitems([request.node])

    assert request.node.get_closest_marker("enable_socket")


def test_a_socket_stays_closed_outside_the_network_mark():
    """
    Asserts that a test without the `network` mark cannot open a connection,
    pytest-socket issuing a warning and then raising `SocketBlockedError` on
    the attempt.

    `create_connection` looks `getaddrinfo` up on the socket module each
    time it runs rather than binding it once at import, and that name is the
    one pytest-socket replaces when a test starts.
    """
    with warns(UserWarning), raises(SocketBlockedError):
        create_connection(("blocked.invalid", 80))


def test_a_unix_socket_stays_open():
    """
    Asserts that a test can open a pair of connected Unix sockets, which
    `--allow-unix-socket` leaves open for a process pool reaching its server
    under the `forkserver` start method.

    `socketpair` builds each end through the `socket` class it looks up
    on the socket module when it runs, which is the class pytest-socket
    replaces, whereas a class bound at import would escape the block and
    pass without the option.
    """
    left, right = socketpair()
    with left, right:
        left.sendall(b"flock")
        assert right.recv(5) == b"flock"


def test_home_is_an_empty_folder():
    """
    Asserts that `~` resolves to an empty folder, so no test reads a dotfile
    of the developer's.
    """
    assert list(Path.home().iterdir()) == []


def test_the_shell_carries_no_variable_that_changes_a_result(name: str):
    """
    Asserts that none of these variables reaches a test, covering the
    ones that set whether a console prints color, the files a GitHub
    Actions runner collects a workflow step's outputs and a workflow run's
    summary page from, and the ones naming where a tool keeps its cache,
    configuration, data, and state.
    """
    assert name not in environ
