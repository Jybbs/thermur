"""
Pins the digest steps the pipeline declares, meaning one per subpackage,
each computing that subpackage's digest and computed again on every run.
"""

from contextlib               import closing
from hamilton.caching.adapter import CachingBehavior
from hamilton.driver          import Builder
from pathlib import Path
from pytest  import Config

from thermur.pipeline         import steps
from thermur.pipeline.graph   import Graph
from thermur.pipeline.schemas import Subpackage
from thermur.repo.checkout    import Checkout


def test_a_digest_step_computes_its_subpackage_s_digest(pytestconfig: Config):
    """
    Asserts that the pipeline's digest step returns the digest of the
    pipeline's own folder beside the `lock_digest` it reads.
    """
    computed = Builder().with_modules(steps).build().execute(
        ["pipeline_digest"],
        inputs = {"lock_digest": "lock"}
    )
    folder = pytestconfig.rootpath / "src/thermur/pipeline"

    assert computed["pipeline_digest"] == Subpackage(folder=folder).digest("lock")


def test_each_subpackage_declares_one_digest_step(
    pytestconfig : Config,
    tmp_path     : Path
):
    """
    Asserts that the steps are one `<subpackage>_digest` per folder
    under `src/thermur`, each computed again on every run, beside the
    `lock_digest` input they read, as the cache resolves each node's
    behavior for a run.
    """
    graph = Graph(
        checkout = Checkout(root=tmp_path),
        modules  = ("thermur.pipeline.steps",)
    )
    with closing(graph):
        graph.driver.execute(["pipeline_digest"], inputs={"lock_digest": "lock"})
    cache   = graph.driver.cache
    folders = (pytestconfig.rootpath / "src/thermur").glob("*/")

    assert cache.behaviors[cache.last_run_id] == dict.fromkeys(
        (f"{folder.name}_digest" for folder in folders),
        CachingBehavior.RECOMPUTE
    ) | {"lock_digest": CachingBehavior.DEFAULT}
