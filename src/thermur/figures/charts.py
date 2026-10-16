"""
Holds `Chart`, one figure a page embeds, which draws under the chart theme
and whose `label` sets a chart's title in the headings' face and its tick
values in the code face, since matplotlib has no rcParam naming a family for
a title or for tick values alone.
"""

from collections.abc   import Iterator
from contextlib        import AbstractContextManager, contextmanager
from matplotlib        import pyplot, rc_context
from matplotlib.axes   import Axes
from matplotlib.figure import Figure
from numpy             import ndarray
from pathlib           import Path
from pydantic          import BaseModel, Field

from thermur.figures.theme import Theme


class Chart(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One figure a page embeds, named for the file it is written to under
    `folder`.
    """

    name: str
    """
    The file the chart is written to, whose suffix sets its format.
    """

    folder: Path = Path("docs/pages/figures")
    """
    The folder the chart is written into, relative to the working directory,
    which a mise task sets to the repository's root.
    """

    theme: Theme = Field(default_factory=Theme)
    """
    The style the chart is drawn in.
    """

    @property
    def path(self) -> Path:
        """
        Joins the folder and the file the chart is written to.
        """
        return self.folder / self.name

    @contextmanager
    def drawn(self, *, ncols: int = 1, nrows: int = 1) -> Iterator[tuple[Figure, Axes | ndarray]]:
        """
        Draws a figure of `nrows` by `ncols` panels under the theme and
        writes it once the block ends, so an error inside the block writes
        nothing.

        Yields:
            The figure beside its one axes, or beside the array of its axes
            where it holds more than one panel.
        """
        with self.themed():
            figure = Figure()
            yield figure, figure.subplots(nrows, ncols)
            self.save(figure)

    def label(
        self,
        ax     : Axes,
        title  : str | None = None,
        xlabel : str | None = None,
        ylabel : str | None = None
    ):
        """
        Sets the title in the headings' face and the tick values in the code
        face, keeping the axis labels in the body text's face, where a label
        left out keeps the text a plotting call wrote.
        """
        families = self.theme.families
        if title is not None:
            ax.set_title(title, fontfamily=families["display"])

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.tick_params(labelfontfamily=families["mono"])

    def save(self, figure: Figure):
        """
        Writes `figure` to the chart's file and closes it, which releases a
        figure seaborn built through `pyplot`.
        """
        self.path.parent.mkdir(exist_ok=True, parents=True)
        figure.savefig(self.path)
        pyplot.close(figure)

    def themed(self) -> AbstractContextManager[None]:
        """
        Builds the context holding the theme in force for its block, so a
        figure seaborn builds itself is drawn in it, and nothing the block
        sets reaches code outside it.
        """
        return rc_context(self.theme.params)
