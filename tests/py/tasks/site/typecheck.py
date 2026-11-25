"""
Pins the single-file component the type check refuses before `tsc` runs.
"""

from common.tasks import run_site
from pathlib      import Path
from pytest       import Config, mark


@mark.parametrize(
    "component",
    [
        ".vitepress/Card.vue", "kit/Card.vue",
        "../tests/kit/Card.vue", "../tests/site/Card.vue"
    ]
)
def test_the_type_check_refuses_a_single_file_component(
    component    : str,
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that the type check fails before `tsc` runs, naming each
    single-file component under `.vitepress`, the kit, or either suite under
    `tests/`, since `tsc` reads none, and leaving one under the kit's own
    packages alone.
    """
    (site / "kit/node_modules").mkdir()
    (site / "kit/node_modules/Packaged.vue").write_text("", encoding="utf-8")
    (site / component).write_text("", encoding="utf-8")
    checked = run_site(site, pytestconfig, "typecheck")

    assert (checked.returncode, checked.stdout) == (1, "")
    assert checked.stderr.rstrip().endswith(component)
