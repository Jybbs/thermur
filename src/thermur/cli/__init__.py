"""
Holds `app`, the `thermur` command line `[project.scripts]` names, which
registers each command by its import path, so `thermur --version` and the
help of a group import no module a command reads until that command runs.
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
app["fetch"].command("thermur.cli.fetch:sample")
app.command("thermur.cli.fly:fly")
