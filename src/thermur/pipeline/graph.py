"""
Holds `Graph`, which runs the Hamilton steps of every subpackage over
one `Settings` through a driver whose cache keeps each result under
`.cache/hamilton/`, beside `Recorder`, which records each run of a recorded
graph under `data/runs/`, and the NetCDF adapters, `NetcdfWriter` saving
each table a run computes and `NetcdfReader` and `NetcdfTreeReader` opening
one. Importing the module registers the adapters with Hamilton, and a
fingerprint reading the type and the values of each array, xarray object,
record, component, and path.
"""

from collections.abc import Mapping, Sequence, Set
from dataclasses     import dataclass
from decimal         import Decimal
from equinox         import Module
from functools       import cached_property
from hamilton        import registry
from hamilton.caching.fingerprinting import (
    UNHASHABLE,
    hash_bytes,
    hash_mapping,
    hash_primitive,
    hash_sequence,
    hash_set,
    hash_value
)
from hamilton.driver                import Builder, Driver
from hamilton.io.data_adapters      import DataLoader, DataSaver
from hamilton.io.materialization    import to
from hamilton.plugins.h_experiments import ExperimentTracker
from importlib import import_module
from jax       import Array
from numpy     import asarray, generic, ndarray
from os        import chdir
from pathlib   import PurePath
from pydantic  import AnyUrl, BaseModel, Field
from uuid      import UUID
from xarray    import DataArray, DataTree, Dataset, open_dataset, open_datatree

from thermur.pipeline.schemas  import Run, Subpackage
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout


@dataclass(frozen=True)
class NetcdfFile:
    """
    A NetCDF-4 file at `path`, which Hamilton's savers and loaders reach
    under the one name `to`, `from_`, and the `format` of `@cache` read.
    """

    path: str
    """
    The file, relative to the working directory, which the experiment
    tracker moves into a run's folder around each saver.
    """

    def __init_subclass__(cls, **kwargs: object):
        """
        Registers each adapter on the file with Hamilton as its class is
        defined, under the `name` the adapters share.
        """
        super().__init_subclass__(**kwargs)
        registry.register_adapter(cls)

    @classmethod
    def name(cls) -> str:
        """
        Names the adapter as `to`, `from_`, and `@cache` read it.
        """
        return "netcdf"


class Recorder(ExperimentTracker):
    """
    The experiment tracker Hamilton ships, recording a `Run` for each
    execution of the graph in place of the configuration it records, in the
    `thermur` experiment.

    The tracker names each run by a random identifier, gives it a folder of
    its own under the experiment's folder, and moves into that folder around
    each saver, so a saver writing a bare file name writes into it.
    """

    def __init__(self, checkout: Checkout, settings: Settings):
        """
        Keeps the `checkout` whose `data/runs/` holds each record and the
        `settings` each execution of the graph records, with no run open.
        """
        self.checkout = checkout
        self.opened   = False
        self.settings = settings

    def run_after_graph_execution(self, **kwargs: object):
        """
        Writes the record of the run `run_before_graph_execution` opened,
        and nothing for an execution that failed before its run opened.

        Hamilton calls this after every execution, a failed one included,
        where the tracker alone raises `AttributeError` before its first
        run and afterward writes the previous run's record again under that
        run's identifier.
        """
        if self.opened:
            self.opened = False
            super().run_after_graph_execution(**kwargs)

    def run_after_node_execution(self, *, success: bool, **kwargs: object):
        """
        Records each saver's file as the tracker does, and after a failed
        node moves back into the folder the run started in and records
        nothing.

        The tracker reads `path` out of a failed saver's result, which is
        `None`, so it raises `TypeError` in place of the saver's own error
        and never moves back out of the run's folder.
        """
        if success:
            super().run_after_node_execution(success=success, **kwargs)
        else:
            chdir(self.init_directory)

    def run_before_graph_execution(self, **kwargs: object):
        """
        Opens a run of its own under the clone's `data/runs/` for each
        execution of the graph, since the tracker fixes its identifier and
        its folder when built, and stores the `Run` read from the clone at
        that moment as the configuration the tracker writes before recording
        the execution's inputs.

        The `Run` holds the settings in full, so the inputs leave out
        `settings`, which the tracker would write as the text `str` gives.
        """
        super().__init__(
            base_directory  = str(self.checkout.root / "data" / "runs"),
            experiment_name = "thermur"
        )
        self.run    = Run.from_checkout(self.checkout, self.settings)
        self.config = self.run.model_dump(mode="json")
        super().run_before_graph_execution(**kwargs)
        self.inputs = [entry for entry in self.inputs if entry.name != "settings"]
        self.opened = True


class Graph(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The Hamilton steps every subpackage declares, the settings they read,
    and the clone their cache and their run record sit in.

    Hamilton calls each step a node and runs a node only once the nodes
    named by its parameters have run. A node reads the settings as
    `settings`, the digest of `uv.lock` as `lock_digest`, and the digest of
    a subpackage's source as `<subpackage>_digest`.

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

        The cache runs every saver on every run and stores nothing of it, so
        a recorded run writes each table it saves into its own folder.
        """
        return (
            Builder()
            .with_modules(*map(import_module, self.modules))
            .with_cache(
                default_saver_behavior = "disable",
                path = self.checkout.root / ".cache" / "hamilton"
            )
            .with_adapters(*[self.recorder] if self.recorded else [])
            .build()
        )

    @cached_property
    def recorder(self) -> Recorder:
        """
        Builds the tracker recording each run of this graph under the
        clone's `data/runs/`.
        """
        return Recorder(self.checkout, self.settings)

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

        Raises:
            ValueError: Where `steps` names no node, or a node the graph
                        does not declare, before the cache or the recorder
                        starts a run.
        """
        if not steps:
            raise ValueError("Name at least one step to compute")

        arguments = {
            "additional_vars" : list(steps),
            "inputs"          : {
                "lock_digest" : self.checkout.lock_digest,
                "settings"    : self.settings
            }
        }
        savers = [
            to.netcdf(dependencies=[step], id=f"{step}_saved", path=f"{step}.nc")
            for step in steps
            if self.recorded and step in self.tables
        ]
        self.driver.validate_materialization(*savers, **arguments)
        _, results = self.driver.materialize(*savers, **arguments)

        return results


class NetcdfReader(NetcdfFile, DataLoader):
    """
    The loader opening the file at `path` as a `Dataset` through h5netcdf.
    """

    @classmethod
    def applicable_types(cls) -> list[type]:
        """
        Lists the type the loader opens.
        """
        return [Dataset]

    def load_data(self, type_: type) -> tuple[Dataset, dict[str, str]]:
        """
        Opens the file at `path`, reading no values until a caller reads
        them.

        Returns:
            The dataset beside its `path` as the loader's metadata, in
            place of the `get_file_metadata` result Hamilton's own loaders
            return, which calls the deprecated `datetime.utcnow`.
        """
        return open_dataset(self.path, engine="h5netcdf"), {"path": self.path}


class NetcdfTreeReader(NetcdfFile, DataLoader):
    """
    The loader opening the file at `path` as a `DataTree` through h5netcdf.

    Hamilton's cache opens a stored result through the loader it pickled
    beside it, calling `load_data` with no type, so each loader opens the
    one type its class lists.
    """

    @classmethod
    def applicable_types(cls) -> list[type]:
        """
        Lists the type the loader opens.
        """
        return [DataTree]

    def load_data(self, type_: type) -> tuple[DataTree, dict[str, str]]:
        """
        Opens the file at `path`, reading no values until a caller reads
        them.

        Returns:
            The tree beside its `path` as the loader's metadata.
        """
        return open_datatree(self.path, engine="h5netcdf"), {"path": self.path}


class NetcdfWriter(NetcdfFile, DataSaver):
    """
    The saver writing a `Dataset` or a `DataTree` into the file at `path`
    through h5netcdf.
    """

    @classmethod
    def applicable_types(cls) -> list[type]:
        """
        Lists the types the saver writes.
        """
        return [Dataset, DataTree]

    def save_data(self, data: Dataset | DataTree) -> dict[str, str]:
        """
        Writes `data` into the file at `path`.

        Returns:
            The `path` the experiment tracker records, in place of the
            `get_file_metadata` result Hamilton's own savers return, which
            calls the deprecated `datetime.utcnow`.
        """
        data.to_netcdf(self.path, engine="h5netcdf")
        return {"path": self.path}


def fingerprint(digest: str, value: object) -> str:
    """
    Hashes `digest` beside the module and the qualified name of `value`'s
    type, so two values of different types never share a fingerprint.

    Hamilton's file store keeps one stored result per fingerprint, so
    a value sharing its fingerprint with a value of another type would
    come back from the cache as that other value. The hash goes through
    `hash_primitive`, which dispatches on no type and so reaches none of the
    registrations below.
    """
    kind = type(value)
    return hash_primitive((kind.__module__, kind.__qualname__, digest))


@hash_value.register(Array)
@hash_value.register(generic)
@hash_value.register(ndarray)
def fingerprint_array(array: Array | generic | ndarray, *, depth: int = 0) -> str:
    """
    Fingerprints a NumPy or a JAX array, or a NumPy scalar, by its type, its
    shape, its dtype, and its values, where Hamilton reads a NumPy array's
    bytes alone, which every array of zeros spanning as many bytes shares
    whatever its shape or dtype, and gives a JAX array and a NumPy integer
    or boolean the one fingerprint it gives an object it cannot read.
    """
    values  = asarray(array)
    payload = (values.shape, values.dtype.str, values.tobytes())
    return fingerprint(hash_value(payload, depth=depth + 1), array)


@hash_value.register(bytes)
def fingerprint_bytes(data: bytes, *, depth: int = 0) -> str:
    """
    Fingerprints bytes by their type and the hash Hamilton takes of the
    bytes themselves, which Hamilton alone gives `b"1.0"` and the number
    `1.0` alike.
    """
    return fingerprint(hash_bytes(data), data)


@hash_value.register(Mapping)
@hash_value.register(Sequence)
@hash_value.register(Set)
def fingerprint_collection(
    collection : Mapping | Sequence | Set,
    *,
    depth      : int = 0
) -> str:
    """
    Fingerprints a mapping, a sequence, or a set by its type and the hash
    Hamilton takes of its items, which Hamilton alone gives a list and a
    tuple, or a set and a `frozenset`, holding the same items.
    """
    match collection:
        case Mapping() : digest = hash_mapping(collection, depth=depth)
        case Set()     : digest = hash_set(collection, depth=depth)
        case _         : digest = hash_sequence(collection, depth=depth)
    return fingerprint(digest, collection)


@hash_value.register(DataArray)
@hash_value.register(Dataset)
def fingerprint_dataset(data: DataArray | Dataset, *, depth: int = 0) -> str:
    """
    Fingerprints a `DataArray` or a `Dataset` by its type and every
    variable's name, dimensions, attributes, and values, which `to_dict`
    lays out as a mapping holding each variable's values as an array, where
    Hamilton reads a `Dataset`'s data variable names alone and nothing of
    a `DataArray`.
    """
    return fingerprint(hash_value(data.to_dict(data="array"), depth=depth + 1), data)


@hash_value.register(Module)
def fingerprint_module(module: Module, *, depth: int = 0) -> str:
    """
    Fingerprints an `equinox.Module` by its type and the value of each of
    its fields, where Hamilton reads the fields alone, which two components
    declaring the same parameters share, and gives a component declaring
    none the one fingerprint it gives an object it cannot read.
    """
    return fingerprint(hash_value(vars(module), depth=depth + 1), module)


@hash_value.register(object)
def fingerprint_object(value: object, *, depth: int = 0) -> str:
    """
    Fingerprints any value no other registration reads by its type and
    the hash Hamilton's own default takes of it, which reads the value's
    `__dict__` or its text, so two classes holding the same fields
    fingerprint apart.

    A value Hamilton's default cannot read keeps `UNHASHABLE`, which the
    cache reads as a result to compute again on every run.
    """
    digest = hash_value.__wrapped__(value, depth=depth)
    return digest if digest == UNHASHABLE else fingerprint(digest, value)


@hash_value.register(BaseModel)
def fingerprint_record(record: BaseModel, *, depth: int = 0) -> str:
    """
    Fingerprints a Pydantic record by its type and the value of each field
    its class declares, where Hamilton reads the record's `__dict__`,
    which holds nothing for a record declaring no field and holds each
    `cached_property` once read.
    """
    return fingerprint(hash_value(record.model_dump(), depth=depth + 1), record)


@hash_value.register(AnyUrl)
@hash_value.register(bool)
@hash_value.register(complex)
@hash_value.register(Decimal)
@hash_value.register(float)
@hash_value.register(int)
@hash_value.register(PurePath)
@hash_value.register(str)
@hash_value.register(UUID)
def fingerprint_text(
    value : AnyUrl | bool | complex | Decimal | float | int | PurePath | str | UUID,
    *,
    depth : int = 0
) -> str:
    """
    Fingerprints a number, a string, a path, a `UUID`, or a URL by its type
    and the text `str` writes for it, which holds the whole value.

    Hamilton reads the text of a `bool`, an `int`, a `float`, or a `str`
    without its type, so `1` and `"1"` share a fingerprint there, as do an
    `IntEnum` member and its value. It gives each other value here the one
    fingerprint it gives an object it cannot read, since it reads a value's
    text only where the value's `__module__` names a module built into the
    interpreter, and a complex number carries no `__module__` at all.
    """
    return fingerprint(hash_primitive(value), value)


@hash_value.register(DataTree)
def fingerprint_tree(tree: DataTree, *, depth: int = 0) -> str:
    """
    Fingerprints a `DataTree` by its type and the `Dataset` at each of
    its paths, where Hamilton reads the names of the tree's variables and
    children alone.
    """
    return fingerprint(hash_value(tree.to_dict(), depth=depth + 1), tree)
