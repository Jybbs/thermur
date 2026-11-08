"""
Holds `app`, the `thermur` command line `[project.scripts]` names, which
registers each command by its import path beside the one line of help a
group lists it by, so `thermur --version` and the help of a group import no
module a command reads until that command runs.
"""

from cyclopts           import App
from importlib.metadata import metadata

app = App(
    help = metadata("thermur")["Summary"],
    name = "thermur"
)
app.command(
    App(
        help = "Fetches the data a run reads into the clone's `data/` folder.",
        name = "fetch"
    )
)
app["fetch"].command(
    "thermur.cli.fetch:sample",
    help = "Fetches the WRF-SFIRE sample into the clone's `data/` folder."
)
app.command(
    "thermur.cli.fly:fly",
    help = "Flies the flock, or computes the steps of the graph named."
)
