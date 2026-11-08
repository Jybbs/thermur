#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "measure"
# MISE description = "Write the trials' means the rules charts draw, out of a run"
# USAGE arg "<run>" help="A run under data/runs holding the coupling and heading trials"
"""
Writes the means and standard errors `thermur.docs.measured` reads out of
the coupling and heading trials a run saved into the file the charts on The
Interaction Rules draw from, as the `docs:measure` task, printing the path
it wrote.
"""

from json    import dumps
from pathlib import Path
from sys     import argv

from thermur.docs.measured import MEASURED, measured


def main():
    """
    Writes the trials of the run the first argument names into `MEASURED`
    and prints its path.
    """
    MEASURED.parent.mkdir(exist_ok=True, parents=True)
    MEASURED.write_text(
        dumps(measured(Path(argv[1])), indent=2) + "\n",
        encoding = "utf-8"
    )
    print(MEASURED)


if __name__ == "__main__":
    main()
