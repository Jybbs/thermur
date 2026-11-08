"""
Declares the controller's Hamilton step, meaning the composite the settings
nest.
"""

from thermur.controller.schemas import Composite
from thermur.pipeline.settings  import Settings


def composite(controller_digest: str, settings: Settings) -> Composite:
    """
    Reads each slot from the settings.
    """
    return Composite.from_settings(settings)
