"""
Holds `Theme`, the style every chart shares, gathered from the palette and
the faces `thermur.identity` declares, whose `params` builds the rcParams
matplotlib and seaborn read once each face the package ships under `fonts/`
is registered with matplotlib.
"""

from functools           import cached_property
from importlib.resources import files
from matplotlib          import cycler, font_manager
from pydantic            import BaseModel, Field

from thermur.identity.schemas import Identity


class Theme(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The style every chart shares, read from the identity's light scheme.
    """

    dpi: int = 200
    """
    The pixels per inch a chart is saved at.
    """

    grid: float = 0.14
    """
    The share of the ink a chart's gridlines are drawn in.
    """

    identity: Identity = Field(default_factory=Identity.shipped)
    """
    The palette and the faces every chart reads.
    """

    plate: tuple[float, float] = (7.0, 4.5)
    """
    The width and the height of every single-panel chart, in inches.
    """

    sizes: tuple[float, float, float] = (10, 11, 14)
    """
    The sizes of the tick values, the axis labels, and the title, in points,
    each a step larger than the last.
    """

    @cached_property
    def families(self) -> dict[str, str]:
        """
        Registers each file every face ships as with matplotlib in place
        of any copy of the same family the machine has installed, since
        `findfont` takes the first entry that matches best, then maps each
        kind of text to its face's family.
        """
        families        = {kind: face.family for kind, face in self.identity.faces}
        manager         = font_manager.fontManager
        manager.ttflist = [
            entry for entry in manager.ttflist if entry.name not in families.values()
        ]
        for _, face in self.identity.faces:
            for font in face.fonts:
                manager.addfont(files(__package__) / "fonts" / font.file)

        return families

    @property
    def params(self) -> dict[str, object]:
        """
        Builds the rcParams every chart is drawn under, setting the body
        face as the default family once each face is registered.
        """
        palette     = self.identity.palette
        ground, ink = palette.hex(palette.ground.light), palette.hex(palette.ink.light)
        tick, label, title = self.sizes
        return {
            "axes.edgecolor"    : ink,
            "axes.facecolor"    : ground,
            "axes.grid"         : True,
            "axes.labelcolor"   : ink,
            "axes.labelsize"    : label,
            "axes.prop_cycle"   : cycler(
                color = [palette.hex(pair.light) for pair in palette.series]
            ),
            "axes.spines.right" : False,
            "axes.spines.top"   : False,
            "axes.titlecolor"   : ink,
            "axes.titlesize"    : title,
            "figure.constrained_layout.use" : True,
            "figure.facecolor"              : ground,
            "figure.figsize"                : self.plate,
            "font.size"       : label,
            "grid.alpha"      : self.grid,
            "grid.color"      : ink,
            "legend.frameon"  : False,
            "savefig.dpi"     : self.dpi,
            "text.color"      : ink,
            "xtick.color"     : ink,
            "xtick.labelsize" : tick,
            "ytick.color"     : ink,
            "ytick.labelsize" : tick,
            "font.family"     : [self.families["body"]]
        }
