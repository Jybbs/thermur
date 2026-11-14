"""
Holds the commands of the `thermur fetch` group, each a plain function the
app registers by its import path, which the `data:fetch` task reaches as
`thermur fetch sample`.
"""

from thermur.environment.schemas import Sample
from thermur.repo.checkout       import Checkout


def sample():
    """
    Fetches the WRF-SFIRE sample into the clone's `data/` folder.

    Downloads the file from the bucket keying it by its pinned SHA-256
    digest, checks it against that digest, and moves it into place, then
    prints the path of the output the graph reads, relative to the clone,
    which the `simulation` setting takes. Exits nonzero, moving nothing,
    where the download fails or the file's digest differs from the pinned
    one.
    """
    root = Checkout.installed().root
    print(Sample().fetch(root).relative_to(root))
