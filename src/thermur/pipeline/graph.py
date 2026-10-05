"""
Holds `Graph`, which runs the Hamilton steps of every subpackage over
one `Settings` through a driver whose cache keeps each result under
`.cache/hamilton/`, beside `Recorder`, which records each run of a recorded
graph under `data/runs/`, and `NetcdfWriter`, which saves each table the
run computes. Importing the module registers with Hamilton the saver and a
fingerprint for each type whose values Hamilton's own fingerprints miss.
"""

from dataclasses import dataclass
from functools   import cached_property
from hamilton    import registry
from hamilton.caching.fingerprinting import hash_value
from hamilton.driver                 import Builder, Driver
from hamilton.io.data_adapters       import DataSaver
from hamilton.io.materialization     import to
from hamilton.plugins.h_experiments  import ExperimentTracker
from importlib import import_module
from jax       import Array
from numpy     import asarray, ndarray
from pathlib   import Path
from pydantic  import BaseModel, Field
from xarray    import DataArray, DataTree, Dataset

from thermur.pipeline.schemas  import Run, Subpackage
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout


@dataclass(frozen=True)
class NetcdfWriter(DataSaver):
    """
    The saver `to.netcdf` names, writing a `Dataset` or a `DataTree` into
    the NetCDF-4 file at `path` through h5netcdf.
    """

    path: str
    """
    The file the saver writes, relative to the folder the experiment tracker
    moves into around each saver.
    """

    @classmethod
    def applicable_types(cls) -> list[type]:
        """
        Lists the types the saver writes.
        """
        return [Dataset, DataTree]

    @classmethod
    def name(cls) -> str:
        """
        Names the saver as `to` reads it.
        """
        return "netcdf"

    def save_data(self, data: Dataset | DataTree) -> dict[str, str]:
        """
        Writes `data` into the file at `path`.

        Returns:
            The `path` the experiment tracker records, in place of the file
            metadata Hamilton's own savers return, which calls the
            deprecated `datetime.utcnow`.
        """
        data.to_netcdf(self.path, engine="h5netcdf")
        return {"path": self.path}


class Recorder(ExperimentTracker):
    """
    The experiment tracker Hamilton ships, recording one `Run` in place of
    the configuration it records, in the `thermur` experiment.

    The tracker names each run by a random identifier, gives it a folder of
    its own under the experiment's folder, and moves into that folder around
    each saver, so a saver writing a bare file name writes into it.
    """

    def __init__(self, folder: Path, run: Run):
        """
        Opens the folder of the run under `folder`, the `data/runs/` of
        the clone.
        """
        super().__init__(base_directory=str(folder), experiment_name="thermur")
        self.run = run

    def run_after_graph_construction(self, **_: object):
        """
        Stores the run's record as the configuration the tracker writes, in
        place of the driver's configuration, which the graph leaves empty.
        """
        self.config = self.run.model_dump(mode="json")


class Graph(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The Hamilton steps every subpackage declares, the settings they read,
    and the clone their cache and their run record sit in.

    Hamilton calls each step a node and runs a node only once the nodes
    named by its parameters have run. A node reads the settings as
    `settings`, the digest of `uv.lock` as `lockfile`, and the digest of a
    subpackage's source as `<subpackage>_digest`.

    The cache keeps each result under a key hashed from the node's own
    source and the fingerprint of each input, so a rerun computes only the
    nodes whose source or inputs changed, and a node reading a digest is
    computed again once the code it calls changes.
    """

    checkout: Checkout = Field(default_factory=Checkout.installed)
    """
    The clone the package was installed from, under which the cache and
    each run's record sit.
    """

    modules: tuple[str, ...] = Field(
        default_factory = lambda: tuple(
            subpackage.steps for subpackage in Subpackage.listed() if subpackage.steps
        )
    )
    """
    The dotted name of each module of steps, every subpackage's `steps`
    module by default.
    """

    recorded: bool = False
    """
    Whether each run records its `Run` under `data/runs/` and saves there
    every table it computes.
    """

    settings: Settings = Field(default_factory=Settings)
    """
    The settings every node reads.
    """

    @cached_property
    def driver(self) -> Driver:
        """
        Builds the Hamilton driver over the steps `modules` declares, with
        its cache under `.cache/hamilton/` and the `recorder` where the
        graph records its runs.

        The cache computes every saver again on every run, so a recorded run
        writes each table it saves into its own folder.
        """
        return (
            Builder()
            .with_modules(*map(import_module, self.modules))
            .with_cache(
                default_saver_behavior = "recompute",
                path = self.checkout.root / ".cache" / "hamilton"
            )
            .with_adapters(*[self.recorder] if self.recorded else [])
            .build()
        )

    @cached_property
    def recorder(self) -> Recorder:
        """
        Opens the record of this graph's runs under the clone's
        `data/runs/`.
        """
        return Recorder(
            self.checkout.root / "data" / "runs",
            Run.from_checkout(self.checkout, self.settings)
        )

    @cached_property
    def tables(self) -> set[str]:
        """
        Names each node returning a type `NetcdfWriter` writes.
        """
        return {
            node.name
            for node in self.driver.list_available_variables()
            if NetcdfWriter.applies_to(node.type)
        }

    def compute(self, *steps: str) -> dict[str, object]:
        """
        Computes the nodes `steps` names and whatever they read, saving
        each table among them into the run's folder where the graph records
        its runs.

        Returns:
            Each result under the name of the node that computed it.
        """
        savers = [
            to.netcdf(dependencies=[step], id=f"{step}_saved", path=f"{step}.nc")
            for step in steps
            if self.recorded and step in self.tables
        ]
        _, results = self.driver.materialize(
            *savers,
            additional_vars = list(steps),
            inputs = {"lockfile": self.checkout.lockfile, "settings": self.settings}
        )

        return results


@hash_value.register(Array)
@hash_value.register(ndarray)
def fingerprint_array(array: Array | ndarray, *, depth: int = 0) -> str:
    """
    Fingerprints a NumPy or a JAX array by its shape, its dtype, and its
    values, where Hamilton reads a NumPy array's bytes alone, which arrays
    of zeros share at every shape, and gives every JAX array the one
    fingerprint it gives an object it cannot read.
    """
    values = asarray(array)
    return hash_value(
        (values.shape, values.dtype.str, values.tobytes()),
        depth = depth + 1
    )


@hash_value.register(DataArray)
@hash_value.register(Dataset)
def fingerprint_dataset(data: DataArray | Dataset, *, depth: int = 0) -> str:
    """
    Fingerprints a `DataArray` or a `Dataset` by every variable's name,
    dimensions, attributes, and values, which `to_dict` lays out as a
    mapping holding each variable's values as an array, where Hamilton reads
    a `Dataset`'s variable names alone and nothing of a `DataArray`.
    """
    return hash_value(data.to_dict(data="array"), depth=depth + 1)


@hash_value.register(BaseModel)
def fingerprint_record(record: BaseModel, *, depth: int = 0) -> str:
    """
    Fingerprints a Pydantic record by the value of each field its class
    declares, where Hamilton reads the record's `__dict__`, which holds
    nothing for a record declaring no field and holds each `cached_property`
    once read.
    """
    return hash_value(record.model_dump(), depth=depth + 1)


@hash_value.register(DataTree)
def fingerprint_tree(tree: DataTree, *, depth: int = 0) -> str:
    """
    Fingerprints a `DataTree` by the `Dataset` at each of its paths, where
    Hamilton reads the tree's variable names alone.
    """
    return hash_value(tree.to_dict(), depth=depth + 1)


registry.register_adapter(NetcdfWriter)
