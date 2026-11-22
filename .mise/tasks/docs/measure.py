#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "measure"
# MISE description = "Write the means the site's charts draw, out of a run"
# USAGE arg "<run>" help="A run under data/runs holding the trials or the sweep"
"""
Writes the means and standard errors `thermur.docs.measured` reads out of
the coupling and heading trials a run saved into the file the charts on The
Interaction Rules draw from, and those it reads out of the sweep a run saved
into the file The Flock in Still Air draws from, as the `docs:measure` task,
printing each path it wrote.
"""

from json    import dumps
from pathlib import Path
from sys     import argv

from thermur.docs.measured import MEASURED, STILL_AIR, measured, swept


def main():
    """
    Writes the trials of the run the first argument names into `MEASURED`
    where the run saved them, and its sweep into `STILL_AIR` where it saved
    one, printing each path written.
    """
    run = Path(argv[1])
    for path, read, table in (
        (MEASURED, measured, "coupling_trial.nc"),
        (STILL_AIR, swept, "sweep.nc")
    ):
        if (run / table).exists():
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(dumps(read(run), indent=2) + "\n", encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
