"""
Pins the controller's step, meaning the composite it reads from the
settings.
"""

from hamilton.driver import Builder

from thermur.controller            import steps
from thermur.controller.components import Constant
from thermur.controller.schemas    import Composite
from thermur.pipeline.settings     import Settings


def test_the_composite_step_reads_each_slot_from_the_settings():
    """
    Asserts that the `composite` step returns the composite the settings
    nest, a slot the settings change included.
    """
    settings = Settings(coupling=Constant(coupling_per_s=1.5))

    computed = Builder().with_modules(steps).build().execute(
        ["composite"],
        inputs = {"controller_digest": "digest", "settings": settings}
    )

    assert computed["composite"] == Composite.from_settings(settings)
    assert computed["composite"].coupling == Constant(coupling_per_s=1.5)
