"""
Pins the graph's contract, covering the fingerprints Hamilton's cache reads
for arrays, datasets, records, components, and paths, the NetCDF saver and
loaders, the modules a graph reads by default, which nodes a rerun reads
back from the cache and as which type, what a new `uv.lock` or a new path
setting computes again, which calls the graph refuses, where the recorder
files a run, and what a recorded run keeps.
"""

from collections.abc import Callable
from common.cached   import Pitch
from common.settings import Heading, Tuned
from decimal         import Decimal
from equinox         import Module
from functools       import partial
from hamilton.caching.adapter        import CachingEventType
from hamilton.caching.fingerprinting import UNHASHABLE, hash_value
from jax            import Array, numpy
from json           import loads
from numpy          import float32, int32, int64, ones, zeros
from pathlib        import Path
from pydantic       import BaseModel
from pytest         import Config, MonkeyPatch, fixture, mark, param, raises
from unittest.mock  import Mock
from uuid           import UUID
from xarray         import DataArray, DataTree, Dataset, open_dataset
from xarray.testing import assert_identical

from thermur.pipeline.graph    import Graph, NetcdfReader, NetcdfTreeReader, NetcdfWriter
from thermur.pipeline.schemas  import Run
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout


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
            Tuned.model_construct(heading=Heading(coupling_per_s=1.0)),
            Tuned.model_construct(heading=Heading(coupling_per_s=2.0)),
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
        param(
            Tuned.model_construct(radius_m=1.0),
            Tuned.model_construct(radius_m=2.0),
            id = "record-values"
        ),
        param(
            Tuned.model_construct(source=Path("a.nc")),
            Tuned.model_construct(source=Path("b.nc")),
            id = "record-path"
        ),
        param(Path("a.nc"), Path("b.nc"), id="path"),
        param(Decimal("1.1"), Decimal("1.2"), id="decimal"),
        param(UUID(int=1), UUID(int=2), id="uuid"),
        param(1 + 2j, 1 + 3j, id="complex"),
        param(int64(3), int64(4), id="numpy-scalar")
    ]
)
def test_values_differing_in_any_part_fingerprint_apart(one: object, other: object):
    """
    Asserts that arrays sharing their bytes but not their shape or dtype,
    and arrays, datasets, trees, records, paths, and scalars differing in a
    value, a dimension, an attribute, or a name, each fingerprint apart.
    """
    assert hash_value(one) != hash_value(other)


@mark.parametrize(
    ("one", "other"),
    [
        param(zeros(3, float32), numpy.zeros(3), id="numpy-jax"),
        param(int64(0), zeros((), int64), id="numpy-scalar-array"),
        param(Heading(), Pitch(), id="components"),
        param(Bare(), Settings.model_construct(), id="records"),
        param(
            Tuned.model_construct(),
            Tuned.model_construct().model_dump(),
            id = "record-dump"
        )
    ]
)
def test_values_of_different_types_fingerprint_apart(one: object, other: object):
    """
    Asserts that two values holding the same contents under different
    types fingerprint apart, so the cache never returns one node's result
    as another's.
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
        param(lambda: Tuned(radius_m=1.0), id="record"),
        param(lambda: Heading(coupling_per_s=2.0), id="component"),
        param(lambda: Path("a.nc"), id="path"),
        param(lambda: int64(3), id="numpy-scalar")
    ]
)
def test_equal_values_fingerprint_alike(made: Callable[[], object]):
    """
    Asserts that two values built alike fingerprint alike, so a rerun reads
    a result back.
    """
    assert hash_value(made()) == hash_value(made())


def test_a_call_naming_no_step_is_refused(graphed: Callable[..., Graph]):
    """
    Asserts that a call naming no step raises a `ValueError` saying so.
    """
    with raises(ValueError, match="Name at least one step to compute"):
        graphed().compute()


@mark.parametrize(
    ("data", "reader"),
    [
        param(
            Dataset({"t": ("x", ones(3))}, attrs={"units": "K"}),
            NetcdfReader,
            id = "dataset"
        ),
        param(
            DataTree.from_dict(
                {"/": Dataset(), "/child": Dataset({"t": ("x", ones(3))})}
            ),
            NetcdfTreeReader,
            id = "tree"
        )
    ]
)
def test_the_saver_writes_a_file_its_loader_reads_back_whole(
    data     : Dataset | DataTree,
    reader   : type[NetcdfReader | NetcdfTreeReader],
    tmp_path : Path
):
    """
    Asserts that the saver writes a dataset or a tree into a NetCDF file
    that the loader for its type opens back identical, each returning the
    path it reached.
    """
    path = str(tmp_path / "saved.nc")

    assert NetcdfWriter(path=path).save_data(data) == {"path": path}
    read, metadata = reader(path=path).load_data(type(data))
    with read:
        assert metadata == {"path": path}
        assert_identical(read, data)


def test_a_failed_save_raises_its_own_error_where_the_run_began(
    clone       : Checkout,
    graphed     : Callable[..., Graph],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a saver failing on a recorded run raises its own error
    and leaves the working directory where the run began rather than in the
    run's folder.
    """
    monkeypatch.chdir(clone.root)
    monkeypatch.setattr(NetcdfWriter, "save_data", Mock(side_effect=OSError("full")))

    with raises(OSError, match="full"):
        graphed(recorded=True).compute("table")

    assert Path.cwd() == clone.root


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


def test_a_misnamed_step_fails_before_any_run_starts(graphed: Callable[..., Graph]):
    """
    Asserts that a step the graph does not declare raises Hamilton's
    unknown-node error before the cache or the recorder starts a run, so the
    cache's last run and the record of the run before it both stand.
    """
    graph = graphed(recorded=True)
    graph.compute("table")
    standing = (
        graph.driver.cache.last_run_id,
        graph.recorder.cache.read(graph.recorder.run_id)
    )

    with raises(ValueError, match=r"Unknown nodes \[tabel\]"):
        graph.compute("tabel")

    assert standing == (
        graph.driver.cache.last_run_id,
        graph.recorder.cache.read(graph.recorder.run_id)
    )


def test_a_new_path_setting_computes_its_reader_again(graphed: Callable[..., Graph]):
    """
    Asserts that a node reading settings differing only in the path a field
    holds is computed again rather than read back from the cache.
    """
    for source in ("a.nc", "b.nc"):
        graph = graphed(settings=Tuned.model_construct(source=Path(source)))
        graph.compute("table")

    assert computed(graph) == {"pipeline_digest", "table"}


def test_a_record_fingerprints_by_its_fields_alone(checkout: Checkout):
    """
    Asserts that a record's fingerprint holds through a read storing a
    `cached_property` beside its fields and moves with the path a field
    holds, and that a record declaring no field still fingerprints.
    """
    checkout.manifest

    assert hash_value(checkout) == hash_value(Checkout(root=checkout.root))
    assert hash_value(checkout) != hash_value(Checkout(root=checkout.root.parent))
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


@mark.parametrize(
    "steps",
    [
        param(("jax_array", "numpy_array"), id="numpy-from-jax"),
        param(("heading", "pitch"), id="components")
    ]
)
def test_a_cached_run_returns_each_node_s_own_type(
    graphed : Callable[..., Graph],
    steps   : tuple[str, ...]
):
    """
    Asserts that two nodes returning the same contents under different types
    each read back their own type from the cache on a rerun.
    """
    made  = graphed(modules=("common.cached",)).compute(*steps)
    graph = graphed(modules=("common.cached",))
    read  = graph.compute(*steps)

    assert computed(graph) == set()
    assert {step: type(read[step]) for step in steps} == {
        step: type(made[step]) for step in steps
    }


@mark.parametrize("step", ["netcdf_table", "netcdf_tree"])
def test_a_netcdf_cached_result_reads_back_from_its_file(
    clone   : Checkout,
    graphed : Callable[..., Graph],
    step    : str
):
    """
    Asserts that a node caching its result as NetCDF reads back identical
    on a rerun, opened from the `.netcdf` file the cache stored under
    `.cache/hamilton/` rather than from a pickle of its values.
    """
    made  = graphed(modules=("common.cached",)).compute(step)[step]
    graph = graphed(modules=("common.cached",))

    with graph.compute(step)[step] as read:
        source = Path(read.encoding["source"])

        assert computed(graph) == set()
        assert_identical(read, made)
        assert (source.parent, source.suffix) == (
            clone.root / ".cache/hamilton",
            ".netcdf"
        )


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
    own, in a folder of their own, holding the `Run` and once each input the
    `Run` does not hold, so the second never replaces the first or lists the
    table the first saved, and the settings sit in the `Run` alone.
    """
    graph = graphed(recorded=True)
    runs  = []
    for step in ("table", "grid"):
        graph.compute(step)
        runs.append(graph.recorder.run_id)
    records = [loads(graph.recorder.cache.read(run)) for run in runs]

    assert len({record["run_dir"] for record in records}) == 2
    assert [len(record["materialized"]) for record in records] == [1, 0]
    assert [[entry["name"] for entry in record["inputs"]] for record in records] == [
        ["lock_digest"]
    ] * 2
    assert [Run.model_validate(record["config"]) for record in records] == [
        graph.recorder.run
    ] * 2


def test_each_run_records_the_lock_digest_as_it_began(
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

    assert record["config"]["lock_digest"] == clone.lock_digest


def test_the_cache_keeps_nothing_of_a_saver(
    clone   : Checkout,
    graphed : Callable[..., Graph]
):
    """
    Asserts that a recorded run stores neither a result nor a metadata row
    for the saver writing its table, so the cache keeps only what nodes
    compute.
    """
    graph = graphed(recorded=True)
    graph.compute("table")
    cache = graph.driver.cache

    assert {row["node_name"] for row in cache.metadata_store.get_run(
        cache.last_run_id
    )} == {
        "pipeline_digest", "table"
    }
    assert not list((clone.root / ".cache/hamilton").glob("table_saved*"))


def test_the_recorder_writes_no_record_for_an_execution_it_never_opened(
    graphed: Callable[..., Graph]
):
    """
    Asserts that an execution failing before the recorder opens its run
    raises Hamilton's own error and writes nothing, before any run and after
    one, so the record of the run before it reads back as written.
    """
    graph  = graphed(recorded=True)
    failed = partial(graph.driver.materialize, additional_vars=["tabel"])

    with raises(ValueError, match="Unknown nodes"):
        failed()
    graph.compute("table")
    record = graph.recorder.cache.read(graph.recorder.run_id)
    with raises(ValueError, match="Unknown nodes"):
        failed()

    assert graph.recorder.cache.read(graph.recorder.run_id) == record
