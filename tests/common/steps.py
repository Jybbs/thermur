"""
Declares the stand-in Hamilton steps the graph's cases run, meaning a table
reading the pipeline's digest and the settings, and an array computed from
that table.
"""

from numpy  import arange, ndarray
from xarray import Dataset

from thermur.pipeline.settings import Settings


def grid(table: Dataset) -> ndarray:
    """
    Doubles the table's values.
    """
    return table.t.values * 2


def table(pipeline_digest: str, settings: Settings) -> Dataset:
    """
    Builds a table of three values, reading the pipeline's digest and the
    settings, so either one changing computes it again.
    """
    return Dataset({"t": ("x", arange(3.0))})
