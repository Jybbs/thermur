"""
Holds the `--share` option, which runs one share of the suite split by test
across several machines, through the hooks adding the option and skipping
every test outside the share it names. The `-p` option in `addopts` loads
this module as a plugin, and `tests/py/shares.py` checks it.
"""

from pytest import Config, Item, Parser, mark


def pytest_addoption(parser: Parser):
    """
    Adds `--share`, which takes `<index>/<count>` and runs the share of the
    suite at `index`, counting from 1, out of `count` shares.
    """
    parser.addoption(
        "--share",
        help    = "Run the share at <index>, counting from 1, out of <count> shares",
        metavar = "<index>/<count>",
        type    = share
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]):
    """
    Skips every test outside the share `--share` names. A share takes every
    `count`-th test from the one at `index` in node ID order, which every
    machine computes alike under any `pytest-randomly` seed. A skip leaves
    syrupy reading every test as selected, so it still reports a snapshot no
    test reads, while it reads a skipped test's snapshots as still in use.
    """
    if picked := config.getoption("share"):
        index, count = picked
        kept         = set(sorted(item.nodeid for item in items)[index - 1 :: count])
        skip         = mark.skip(reason=f"Runs in another share than {index}/{count}")
        for item in items:
            if item.nodeid not in kept:
                item.add_marker(skip)


def share(text: str) -> tuple[int, int]:
    """
    Reads `<index>/<count>` as the share at `index` out of `count` shares.

    Raises:
        ValueError: Where `text` is no such pair, or `index` falls outside 1
                    to `count`.
    """
    index, count = map(int, text.split("/"))
    if not 1 <= index <= count:
        raise ValueError(f"No share {index} out of {count}")

    return index, count
