"""
Holds `Tuned`, the package's settings widened by a stand-in field and the
stand-in `Heading` component, which the settings and the graph cases read.
"""

from equinox  import Module
from pydantic import Field
from typing   import Annotated

from thermur.pipeline.settings import Settings


class Heading(Module):
    """
    A stand-in component of the controller, holding one bounded parameter.
    """

    coupling_per_s: Annotated[float, Field(gt=0, le=20)] = 1.0
    """
    The rate at which a drone turns toward its neighbors, in reciprocal
    seconds.
    """


class Tuned(Settings):
    """
    The settings with a stand-in component and a stand-in field.
    """

    heading: Heading = Heading()
    """
    The stand-in component.
    """

    radius_m: Annotated[float, Field(gt=0)] = 5.0
    """
    A stand-in distance in meters.
    """
