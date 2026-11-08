"""
Holds `fly`, the command `thermur fly` runs and the tasks computing a step
of the graph wrap.
"""

from cyclopts import Parameter
from typing   import Annotated

from thermur.pipeline.graph    import Graph
from thermur.pipeline.settings import Settings


def fly(
    *steps   : Annotated[str, Parameter(help="Each step of the graph to compute.")],
    settings : Annotated[Settings | None, Parameter(name="*")] = None
):
    """
    Flies the flock through the graph, recording the run under `data/runs/`.

    With no step named, flies the reference flight through the parametric
    threat, or through the output `--simulation` names, then saves the
    flight's frames and statistics and prints the statistics. With step
    names, computes those steps and whatever they read, saving each table
    among them and printing each. Exits nonzero, computing nothing, where a
    name matches no step, and with status 2 where a flag fails its bound.
    """
    results = Graph(recorded=True, settings=settings or Settings()).compute(
        *steps or ("frames", "statistics")
    )
    for name in steps or ("statistics",):
        print(results[name])
