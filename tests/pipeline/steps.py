"""
Pins the digest steps the pipeline declares, meaning one per subpackage,
each computing that subpackage's digest and computed again on every run.
"""

from hamilton.driver import Builder
from pytest          import Config

from thermur.pipeline         import steps
from thermur.pipeline.schemas import Subpackage


def test_a_digest_step_computes_its_subpackage_s_digest(pytestconfig: Config):
    """
    Asserts that the pipeline's digest step returns the digest of the
    pipeline's own folder beside the `lockfile` it reads.
    """
    computed = Builder().with_modules(steps).build().execute(
        ["pipeline_digest"],
        inputs = {"lockfile": "lock"}
    )
    folder = pytestconfig.rootpath / "src/thermur/pipeline"

    assert computed["pipeline_digest"] == Subpackage(folder=folder).digest("lock")


def test_each_subpackage_declares_one_digest_step(pytestconfig: Config):
    """
    Asserts that the steps are one `<subpackage>_digest` per folder under
    `src/thermur`, each computed again on every run, beside the `lockfile`
    input they read.
    """
    behaviors = {
        node.name: node.tags.get("cache.behavior")
        for node in (
            Builder().with_modules(steps)
                     .build()
                     .list_available_variables()
        )
    }
    folders = (pytestconfig.rootpath / "src/thermur").glob("*/")

    assert behaviors == dict.fromkeys(
        (f"{folder.name}_digest" for folder in folders),
        "recompute"
    ) | {"lockfile": None}
