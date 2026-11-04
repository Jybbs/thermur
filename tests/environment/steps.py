"""
Pins the fire data's steps, meaning the output the settings choose, read
again on every run, the air the graph lays out from it, and when a rerun
reads that air back from the cache.
"""

from common.runs              import computed
from hamilton.caching.adapter import CachingBehavior
from numpy.testing            import assert_allclose, assert_array_equal
from pathlib import Path
from pytest  import fixture

from thermur.environment.air     import Air
from thermur.environment.schemas import Crop, Grid, ParametricThreat
from thermur.environment.wrf     import WrfOutput
from thermur.pipeline.graph      import Graph
from thermur.pipeline.settings   import Settings
from thermur.repo.checkout       import Checkout


@fixture
def settings() -> Settings:
    """
    Builds settings reading the threat on a small grid, which the crop's
    fixed heights from 20 m to 100 m sit inside.
    """
    return Settings(
        crop   = Crop(floor_m=20.0, levels=5, spacing_m=20.0),
        threat = ParametricThreat(
            edge_m = 10.0,
            grid   = Grid(
                columns   = 8,
                levels    = 10,
                rows      = 2,
                snapshots = 2
            )
        )
    )


def graphed(clone: Checkout, settings: Settings) -> Graph:
    """
    Builds a graph over the fire data's steps and the pipeline's, each graph
    a fresh run reading the cache the runs before it left in `clone`.
    """
    return Graph(
        checkout = clone,
        modules  = ("thermur.pipeline.steps", "thermur.environment.steps"),
        settings = settings
    )


def test_a_second_run_reads_the_air_back_from_the_cache(
    clone    : Checkout,
    settings : Settings
):
    """
    Asserts that a second run of one set of settings opens the output again
    and reads back the `Air` the first run laid out, its channels and its
    grid as the first run left them.
    """
    first = graphed(clone, settings).compute("air")["air"]
    graph = graphed(clone, settings)

    second = graph.compute("air")["air"]

    assert computed(graph) == {"environment_digest", "wrf_output"}
    assert isinstance(second, Air)
    assert (second.names, second.origin, second.spacing) == (
        first.names, first.origin, first.spacing
    )
    assert_array_equal(second.channels, first.channels)


def test_an_output_changed_on_disk_lays_its_air_out_again(
    clone    : Checkout,
    settings : Settings,
    tmp_path : Path
):
    """
    Asserts that a run reads the air back from the cache while the file at
    the path the settings name holds the same values, and lays it out again
    once the file at that path holds other ones.
    """
    path    = tmp_path / "wrfout.nc"
    dataset = settings.threat.dataset
    dataset.to_netcdf(path, engine="netcdf4")
    settings = settings.model_copy(update={"simulation": path})
    first    = graphed(clone, settings).compute("air")["air"]

    unchanged = graphed(clone, settings)
    unchanged.compute("air")

    dataset.assign(T=dataset.T + 1).to_netcdf(tmp_path / "next.nc", engine="netcdf4")
    (tmp_path / "next.nc").replace(path)
    changed = graphed(clone, settings)
    second  = changed.compute("air")["air"]

    assert "air" not in computed(unchanged)
    assert "air" in computed(changed)
    temperature = first.names.index("temperature")
    assert (second.channels[temperature] > first.channels[temperature]).all()


def test_the_graph_lays_out_the_air_of_the_output_the_settings_choose(
    clone    : Checkout,
    settings : Settings
):
    """
    Asserts that the `air` step returns the air of the output the settings
    choose, here the threat's, and that the output it reads is opened again
    on every run while the air reads the environment's digest.
    """
    graph = graphed(clone, settings)

    air      = graph.compute("air")["air"]
    expected = WrfOutput.from_settings(settings).air

    assert air.names == expected.names
    assert_allclose(air.channels, expected.channels)
    cache = graph.driver.cache
    assert cache.behaviors[cache.last_run_id]["wrf_output"] is CachingBehavior.RECOMPUTE
    assert "environment_digest" in graph.driver.graph.nodes["air"].input_types
