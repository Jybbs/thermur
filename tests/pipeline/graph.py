"""
Pins the graph's contract, covering the fingerprints Hamilton's cache reads
for arrays, datasets, and records, the NetCDF saver, the modules a graph
reads by default, which nodes a rerun reads back from the cache, what a
new `uv.lock` computes again, where the recorder files a run, and what a
recorded run keeps.
"""

from collections.abc import Callable
from common.settings import Heading, Tuned
from equinox         import Module
from functools       import partial
from hamilton.caching.adapter        import CachingEventType
from hamilton.caching.fingerprinting import UNHASHABLE, hash_value
from jax            import Array, numpy
from json           import loads
from numpy          import float32, int32, ones, zeros
from pathlib        import Path
from pydantic       import BaseModel
from pytest         import Config, MonkeyPatch, fixture, mark, param
from xarray         import DataArray, DataTree, Dataset, open_dataset, open_datatree
from xarray.testing import assert_identical

from thermur.pipeline.graph   import Graph, NetcdfWriter
from thermur.pipeline.schemas import Run
from thermur.repo.checkout    import Checkout


class Bare(BaseModel):
    """
    A record declaring no field.
    """


class Holder(Module):
    """
    A record of the numerical core holding one JAX array.
    """

    values: Array
    """
    The array the record holds.
    """


@fixture
def graphed(clone: Checkout) -> Callable[..., Graph]:
    """
    Binds the clone and the modules holding the pipeline's digests and
    the stand-in steps, so a case builds a graph naming only the fields
    it varies.
    """
    return partial(
        Graph,
        checkout = clone,
        modules  = ("thermur.pipeline.steps", "common.steps")
    )


def computed(graph: Graph) -> set[str]:
    """
    Names each node the graph's most recent run computed rather than read
    back from its cache.
    """
    cache = graph.driver.cache
    return {
        node
        for node, events in cache.logs(cache.last_run_id, level="debug").items()
        if any(event.event_type is CachingEventType.EXECUTE_NODE for event in events)
    }


@mark.parametrize(
    ("one", "other"),
    [
        param(zeros((2, 3)), zeros((3, 2)), id="numpy-shape"),
        param(zeros(3, int32), zeros(3, float32), id="numpy-dtype"),
        param(numpy.zeros(3), numpy.ones(3), id="jax-values"),
        param(numpy.zeros((2, 3)), numpy.zeros((3, 2)), id="jax-shape"),
        param(
            Dataset({"t": ("x", zeros(3))}),
            Dataset({"t": ("x", ones(3))}),
            id = "dataset-values"
        ),
        param(DataArray(zeros(3)), DataArray(ones(3)), id="data-array-values"),
        param(
            Dataset({"t": ("x", zeros(3))}),
            Dataset({"t": ("y", zeros(3))}),
            id = "dataset-dimensions"
        ),
        param(
            Dataset({"t": ("x", zeros(3))}, attrs={"units": "K"}),
            Dataset({"t": ("x", zeros(3))}, attrs={"units": "C"}),
            id = "dataset-attributes"
        ),
        param(
            DataArray(zeros(3), name="t"),
            DataArray(zeros(3), name="u"),
            id = "data-array-name"
        ),
        param(
            DataTree.from_dict({"/child": Dataset({"t": ("x", zeros(3))})}),
            DataTree.from_dict({"/child": Dataset({"t": ("x", ones(3))})}),
            id = "tree-child-values"
        ),
        param(
            Tuned(heading=Heading(coupling_per_s=1.0)),
            Tuned(heading=Heading(coupling_per_s=2.0)),
            id = "record-component"
        ),
        param(
            DataTree(Dataset({"t": ("x", zeros(3))})),
            DataTree(Dataset({"t": ("x", ones(3))})),
            id = "tree-values"
        ),
        param(
            DataTree(Dataset({"t": ("x", zeros(3))}, attrs={"units": "K"})),
            DataTree(Dataset({"t": ("x", zeros(3))}, attrs={"units": "C"})),
            id = "tree-attributes"
        ),
        param(Holder(numpy.zeros(3)), Holder(numpy.ones(3)), id="module-values"),
        param(Tuned(radius_m=1.0), Tuned(radius_m=2.0), id="record-values")
    ]
)
def test_values_differing_in_any_part_fingerprint_apart(one: object, other: object):
    """
    Asserts that arrays sharing their bytes but not their shape or dtype,
    and arrays, datasets, trees, and records differing in a value, a
    dimension, an attribute, or a name, each fingerprint apart.
    """
    assert hash_value(one) != hash_value(other)


@mark.parametrize(
    "made",
    [
        param(lambda: zeros((2, 3)), id="numpy"),
        param(lambda: numpy.arange(3.0), id="jax"),
        param(
            lambda: Dataset({"t": ("x", zeros(3))}, attrs={"units": "K"}),
            id = "dataset"
        ),
        param(lambda: DataTree(Dataset({"t": ("x", zeros(3))})), id="tree"),
        param(lambda: Tuned(radius_m=1.0), id="record")
    ]
)
def test_equal_values_fingerprint_alike(made: Callable[[], object]):
    """
    Asserts that two values built alike fingerprint alike, so a rerun reads
    a result back.
    """
    assert hash_value(made()) == hash_value(made())


def test_a_graph_reads_every_subpackage_s_steps_by_default(
    clone        : Checkout,
    pytestconfig : Config
):
    """
    Asserts that a graph built with no modules named reads the steps module
    of every subpackage holding a `steps.py`, in name order.
    """
    steps = sorted((pytestconfig.rootpath / "src/thermur").glob("*/steps.py"))

    assert Graph(checkout=clone).modules == tuple(
        f"thermur.{path.parent.name}.steps" for path in steps
    )


@mark.parametrize(
    ("data", "opened"),
    [
        param(
            Dataset({"t": ("x", ones(3))}, attrs={"units": "K"}),
            open_dataset,
            id = "dataset"
        ),
        param(
            DataTree.from_dict(
                {"/": Dataset(), "/child": Dataset({"t": ("x", ones(3))})}
            ),
            open_datatree,
            id = "tree"
        )
    ]
)
def test_the_saver_writes_a_file_that_reads_back_whole(
    data     : Dataset | DataTree,
    opened   : Callable,
    tmp_path : Path
):
    """
    Asserts that the saver writes a dataset or a tree into a NetCDF file
    that reads back identical, and returns the path it wrote.
    """
    path = tmp_path / "saved.nc"

    assert NetcdfWriter(path=str(path)).save_data(data) == {"path": str(path)}
    with opened(path, engine="h5netcdf") as read:
        assert_identical(read, data)


def test_a_record_fingerprints_by_its_fields_alone(checkout: Checkout):
    """
    Asserts that a record's fingerprint holds through a read storing a
    `cached_property` beside its fields, and that a record declaring no
    field still fingerprints.
    """
    checkout.manifest

    assert hash_value(checkout) == hash_value(Checkout(root=checkout.root))
    assert hash_value(Bare()) != UNHASHABLE


def test_a_recorded_run_keeps_its_record_beside_each_table_it_computes(
    clone   : Checkout,
    graphed : Callable[..., Graph]
):
    """
    Asserts that a recorded run stores its `Run` as the tracker's
    configuration and saves each table it computes into its own folder,
    leaving an array unsaved.
    """
    graph   = graphed(recorded=True)
    results = graph.compute("grid", "table")
    folder  = graph.recorder.run_directory
    record  = loads(graph.recorder.cache.read(graph.recorder.run_id))

    assert Run.model_validate(record["config"]) == graph.recorder.run
    assert [entry["path"] for entry in record["materialized"]] == [
        str(folder / "table.nc")
    ]
    assert folder.parent == clone.root / "data" / "runs" / "thermur"
    with open_dataset(folder / "table.nc", engine="h5netcdf") as saved:
        assert_identical(saved, results["table"])


def test_an_unrecorded_run_saves_nothing(
    clone       : Checkout,
    graphed     : Callable[..., Graph],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a graph that records nothing leaves the clone's `data/`
    untouched and saves no table into the working directory either.
    """
    monkeypatch.chdir(clone.root)
    graphed().compute("grid", "table")

    assert not (clone.root / "data").exists()
    assert not list(clone.root.rglob("*.nc"))


def test_each_recorded_run_saves_its_tables_again(graphed: Callable[..., Graph]):
    """
    Asserts that a recorded run writes its tables into its own folder even
    where the cache holds every result, so no run's folder misses a table.
    """
    for _ in range(2):
        graph = graphed(recorded=True)
        graph.compute("table")

        assert (graph.recorder.run_directory / "table.nc").is_file()


def test_each_run_of_one_graph_keeps_a_record_of_its_own(graphed: Callable[..., Graph]):
    """
    Asserts that two runs of one recorded graph each keep a record of their
    own, in a folder of their own, holding the `Run` and each input once, so
    the second never replaces the first or lists the table the first saved.
    """
    graph = graphed(recorded=True)
    runs  = []
    for step in ("table", "grid"):
        graph.compute(step)
        runs.append(graph.recorder.run_id)
    records = [loads(graph.recorder.cache.read(run)) for run in runs]

    assert len({record["run_dir"] for record in records}) == 2
    assert [len(record["materialized"]) for record in records] == [1, 0]
    assert [len(record["inputs"]) for record in records] == [2, 2]
    assert [Run.model_validate(record["config"]) for record in records] == [
        graph.recorder.run
    ] * 2


def test_each_run_records_the_lockfile_as_it_began(
    clone   : Checkout,
    graphed : Callable[..., Graph]
):
    """
    Asserts that a run's record carries the digest of `uv.lock` as it stood
    when the run began, so a later run of the same graph records a new
    `uv.lock` rather than the first run's.
    """
    graph = graphed(recorded=True)
    graph.compute("table")
    (clone.root / "uv.lock").write_text("version = 1\n")
    graph.compute("table")
    record = loads(graph.recorder.cache.read(graph.recorder.run_id))

    assert record["config"]["lockfile"] == clone.lockfile


@mark.parametrize(
    ("lockfile", "recomputed"),
    [
        param(None, {"pipeline_digest"}, id="unchanged"),
        param("version = 1\n", {"pipeline_digest", "table"}, id="new-lockfile")
    ]
)
def test_a_rerun_computes_again_only_what_its_inputs_change(
    clone      : Checkout,
    graphed    : Callable[..., Graph],
    lockfile   : str | None,
    recomputed : set[str]
):
    """
    Asserts that a second graph over the same clone reads each node back
    from the cache under `.cache/hamilton/`, computing again only the digest
    it reads, and that a new `uv.lock` computes again the node reading that
    digest as well. The node below it still reads its result back, since
    the table it reads comes out unchanged and a table fingerprints by its
    values.
    """
    graphed().compute("grid")
    if lockfile:
        (clone.root / "uv.lock").write_text(lockfile)
    graph = graphed()
    graph.compute("grid")

    assert computed(graph) == recomputed
    assert (clone.root / ".cache/hamilton").is_dir()
