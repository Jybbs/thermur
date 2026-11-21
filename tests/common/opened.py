"""
Holds `Opened`, the audit hook gathering each tracked file other than
Python a test opens in its own process, which the `opens` fixture adds while
pytest-testmon collects and the `recorded` fixture calls `record` on as each
test ends, so a change to that file selects the test, beside the fixture
setup hook `Opened.added` registers this module for as a plugin.
"""

from collections.abc import Iterable, Iterator
from functools       import cache
from os              import PathLike, fsdecode
from pathlib         import Path
from pydantic        import BaseModel
from pytest          import Config, FixtureDef, FixtureRequest, StashKey, hookimpl
from sys             import addaudithook
from testmon.process_code import read_source_sha
from types                import CodeType
from typing               import Self


class Opened(BaseModel, extra="forbid", use_attribute_docstrings=True):
    """
    The audit hook recording each tracked file other than Python the running
    test opens, under its path in the checkout where the test opens a copy.
    """

    tracked: frozenset[Path]
    """
    Each file git tracks that `fingerprinted` holds, by its absolute path.
    """

    copies: dict[Path, Path] = {}
    """
    The root of the checkout each copy a test opens was made from, by the
    copy's own root.
    """

    fixtures: dict[str, set[Path]] = {}
    """
    Each tracked file the setup of a fixture opened, by the fixture's name,
    which every test requesting that fixture counts as opening.
    """

    paused: bool = False
    """
    Whether an open records nothing, as while a copy is made, since copying
    a file reads none of what a test checks.
    """

    pending: set[Path] = set()
    """
    Each tracked file the running test opened, which `record` runs a line
    of, since CPython traces no line an audit hook runs.
    """

    def __call__(self, event: str, args: tuple[object, ...], /):
        """
        Adds to `pending` the file an `open` event names, where it is one
        of `tracked` or the copy of one under a root of `copies`, unless
        `paused` holds.
        """
        if event != "open" or self.paused or not isinstance(args[0], str | PathLike):
            return

        path = Path(fsdecode(args[0])).absolute()
        for copy, root in self.copies.items():
            if path.is_relative_to(copy):
                path = root / path.relative_to(copy)
        if path in self.tracked:
            self.pending.add(path)

    @classmethod
    def added(cls, config: Config, tracked: Iterable[Path]) -> Self:
        """
        Builds the hook over each of `tracked`, relative to the root of
        `config`, that `fingerprinted` holds, adding it as an audit hook
        and stashing it on `config` under `OPENED` where pytest-testmon
        collects, registering this module as the plugin whose fixture setup
        hook reads it. Under any other run the hook tracks no file and is
        never added.

        Returns:
            The hook, which a fixture pauses and points at a copy.
        """
        if not config.pluginmanager.has_plugin("TestmonCollect"):
            return cls(tracked=frozenset())

        root = config.rootpath
        hook = cls(
            tracked = frozenset(
                root / path for path in tracked if fingerprinted(path, root)
            )
        )
        addaudithook(hook)
        config.stash[OPENED] = hook
        config.pluginmanager.import_plugin(__name__)
        return hook

    def record(self, fixtures: Iterable[str]):
        """
        Runs one line compiled under the name of each file of `pending` and
        each file `self.fixtures` holds under a name of `fixtures`, so the
        coverage pytest-testmon collects counts the file as one the running
        test read, then empties `pending`. pytest-testmon fingerprints
        a file other than Python whole, so a change anywhere in the file
        selects the test.
        """
        for path in self.pending.union(
            *(self.fixtures.get(name, ()) for name in fixtures)
        ):
            exec(compiled(path), {})
        self.pending.clear()


OPENED = StashKey[Opened]()
"""
The key `Opened.added` stashes the hook under on the pytest configuration.
"""


@cache
def compiled(path: Path) -> CodeType:
    """
    Compiles one line holding `None` under the name of `path`.

    Returns:
        The code object, which coverage traces as a line of `path`.
    """
    return compile("None", path, "exec")


def fingerprinted(path: Path, root: Path) -> bool:
    """
    Reports whether pytest-testmon fingerprints `path`, relative to `root`,
    as one block of text, which it does for a file other than Python whose
    path holds a dot and whose encoding its own reader detects in the
    first two lines, decoding the rest with each undecodable byte replaced,
    whereas it fails on the record of a test reading any other file and
    drops that record.
    """
    if path.suffix == ".py" or "." not in path.as_posix():
        return False
    try:
        read_source_sha(root / path)
    except SyntaxError:
        return False
    return True


@hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef : FixtureDef,
    request    : FixtureRequest
) -> Iterator[object]:
    """
    Adds to `Opened.fixtures` under the name of `fixturedef` each tracked
    file its setup opens, once `OPENED` is stashed, which `pending` also
    keeps for the running test, so every test requesting a fixture set
    up once counts the files it opened. A plugin module's hook runs for a
    fixture of every scope, whereas a conftest's runs only for a fixture
    whose node sits under the conftest's folder.
    """
    opens = request.config.stash.get(OPENED, None)
    if opens is None:
        return (yield)
    outer, opens.pending = opens.pending, set()
    try:
        return (yield)
    finally:
        opens.fixtures.setdefault(fixturedef.argname, set()).update(opens.pending)
        opens.pending |= outer
