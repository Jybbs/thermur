"""
Holds the commands of the `thermur cut` group, each a plain function the
app registers by its import path, of which `thermur cut sample` writes the
sample `thermur fetch sample` downloads.
"""

from cyclopts import Parameter
from typing   import Annotated

from thermur.environment.schemas import Sample
from thermur.environment.wrf     import WrfOutput
from thermur.pipeline.settings   import Settings
from thermur.repo.checkout       import Checkout


def sample(
    source: Annotated[
        str,
        Parameter(
            help = (
                "The full output the sample is cut from, such as `wrfout_W5F7R4`, "
                "as a path or as the HTTPS address the dataset serves it at."
            )
        )
    ]
):
    """
    Cuts the WRF-SFIRE sample into the clone's `data/` folder.

    Crops the output at `source` to the snapshots the sample pins under
    the crop the graph reads, reading only the bytes the crop covers where
    `source` is an address, checks that the window holds a developed plume
    at the figures the sample pins, and writes it as one NetCDF-4 file
    compressed with zlib, then prints the path of the file relative to the
    clone. Exits nonzero, writing nothing, where `source` names another
    output than the one the sample is cut from, or where the window misses a
    figure it is held to, naming each one it misses.
    """
    root, pinned = Checkout.installed().root, Sample()
    output       = WrfOutput.from_sample(source, sample=pinned, settings=Settings())
    print(output.write(root / pinned.path, sample=pinned).relative_to(root))
