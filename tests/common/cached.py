"""
Declares the stand-in Hamilton steps the cache cases run, meaning a JAX
array beside the NumPy array converted from it, two components holding the
same parameter, a list and a tuple holding the same items, and a table and a
tree the cache stores as NetCDF files.
"""

from common.settings import Heading
from equinox         import Module
from hamilton.function_modifiers import cache
from jax    import Array, numpy
from numpy  import arange, asarray, ndarray
from xarray import DataTree, Dataset


class Pitch(Module):
    """
    A stand-in component declaring the one parameter `Heading` declares,
    under the same default.
    """

    coupling_per_s: float = 1.0
    """
    The rate at which a drone levels toward its neighbors, in reciprocal
    seconds.
    """


@cache(format="netcdf")
def netcdf_table() -> Dataset:
    """
    Builds a table of three values, which the cache stores as a NetCDF file.
    """
    return Dataset({"t": ("x", arange(3.0))}, attrs={"units": "K"})


def heading() -> Heading:
    """
    Builds the stand-in heading component at its default.
    """
    return Heading()


def jax_array() -> Array:
    """
    Builds a JAX array of three values.
    """
    return numpy.arange(3.0)


def numpy_array(jax_array: Array) -> ndarray:
    """
    Converts the JAX array into a NumPy array holding the same values.
    """
    return asarray(jax_array)


@cache(format="netcdf")
def netcdf_tree() -> DataTree:
    """
    Builds a tree holding a table of three values under one child, which the
    cache stores as a NetCDF file.
    """
    return DataTree.from_dict(
        {"/": Dataset(), "/n3": Dataset({"t": ("x", arange(3.0))})}
    )


def pair_list() -> list[int]:
    """
    Builds a list of two numbers.
    """
    return [1, 2]


def pair_tuple() -> tuple[int, int]:
    """
    Builds a tuple holding the numbers `pair_list` holds.
    """
    return (1, 2)


def pitch() -> Pitch:
    """
    Builds the stand-in pitch component at its default.
    """
    return Pitch()
