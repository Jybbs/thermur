"""
Declares the fire data's Hamilton steps, meaning the WRF-SFIRE output the
settings name, or the dataset the parametric threat builds where they name
none, and the air a flight samples from it.
"""

from hamilton.function_modifiers import cache

from thermur.environment.air   import Air
from thermur.environment.wrf   import WrfOutput
from thermur.pipeline.settings import Settings


def air(environment_digest: str, wrf_output: WrfOutput) -> Air:
    """
    Lays out the output's channels on the uniform grid a flight samples.
    """
    return wrf_output.air


@cache(behavior="recompute")
def wrf_output(settings: Settings) -> WrfOutput:
    """
    Opens the output the settings name, cropped before any other field
    loads, reading only the times and the base-state geopotential, or wraps
    the dataset the threat builds, again on every run, since the cache reads
    the path of a file rather than what the file holds.
    """
    return WrfOutput.from_settings(settings)
