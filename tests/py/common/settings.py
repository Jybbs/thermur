"""
Holds `Tuned`, the package's settings widened by a stand-in field and a
stand-in path, which the settings and the graph cases read.
"""

from pathlib  import Path
from pydantic import Field
from typing   import Annotated

from thermur.pipeline.settings import Settings


class Tuned(Settings):
    """
    The settings with a stand-in field and a stand-in path.
    """

    radius_m: Annotated[float, Field(gt=0)] = 5.0
    """
    A stand-in distance in meters.
    """

    source: Path = Path("data/raw/sample.nc")
    """
    A stand-in file a step reads.
    """
