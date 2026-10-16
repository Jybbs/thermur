"""
Pins what `Chart` writes and how it finishes a chart, meaning the file drawn
under the theme once the block ends, nothing written where the block fails,
the faces its title and tick values take, and a figure seaborn builds drawn
in the theme's series.
"""

from matplotlib import pyplot
from pathlib    import Path
from pytest     import mark, param, raises
from seaborn    import barplot, color_palette

from thermur.figures.charts import Chart


def test_a_chart_failing_inside_its_block_writes_nothing(tmp_path: Path):
    """
    Asserts that an error raised inside the block propagates and leaves no
    file behind.
    """
    chart = Chart(folder=tmp_path, name="failed.png")

    with raises(ValueError), chart.drawn():
        raise ValueError("drawn wrong")

    assert not chart.path.exists()


def test_a_chart_of_several_panels_yields_an_array_of_axes(tmp_path: Path):
    """
    Asserts that a chart of two rows and two columns yields its four axes.
    """
    with Chart(folder=tmp_path, name="grid.png").drawn(ncols=2, nrows=2) as (_, axes):
        assert axes.shape == (2, 2)


def test_a_drawn_chart_writes_its_file_under_its_folder(tmp_path: Path):
    """
    Asserts that a chart drawn through `drawn` writes a PNG at the plate's
    size and the theme's resolution under its folder once the block ends.
    """
    chart = Chart(folder=tmp_path / "figures", name="flight.png")

    with chart.drawn() as (figure, ax):
        ax.plot([0, 1], [0, 1])
        assert figure.get_size_inches().tolist() == list(chart.theme.plate)

    png = chart.path.read_bytes()
    assert png[1:4] == b"PNG"
    assert int.from_bytes(png[16:20]) == chart.theme.plate[0] * chart.theme.dpi


def test_a_figure_seaborn_builds_takes_the_theme_s_series(tmp_path: Path):
    """
    Asserts that seaborn reads the theme's color cycle inside `themed`, so
    a chart takes the identity's series with no palette of its own, and that
    saving the figure seaborn built through `pyplot` closes it.
    """
    chart = Chart(folder=tmp_path, name="seaborn.png")

    with chart.themed():
        assert color_palette().as_hex()[0] == "#bb7733"
        figure = barplot(x=["a", "b"], y=[1, 2]).figure
        chart.save(figure)

    assert chart.path.exists()
    assert figure.number not in pyplot.get_fignums()


@mark.parametrize("xlabel", [param("Frame", id="given"), param(None, id="left-out")])
def test_a_label_sets_the_title_and_the_ticks_in_their_faces(
    tmp_path : Path,
    xlabel   : str | None
):
    """
    Asserts that the title takes the display face and the tick values the
    mono face, where an axis label given replaces the text a plotting call
    wrote and one left out keeps it.
    """
    chart = Chart(folder=tmp_path, name="labeled.png")

    with chart.drawn() as (_, ax):
        ax.set_xlabel("Written")
        chart.label(ax, title="Polarization", xlabel=xlabel, ylabel="Polarization")

        assert ax.title.get_fontfamily() == ["Bellefair"]
        assert ax.get_xticklabels()[0].get_fontfamily() == ["JetBrains Mono"]
        assert (
            ax.get_xlabel(),
            ax.get_ylabel()
        ) == (xlabel or "Written", "Polarization")


def test_a_label_naming_nothing_keeps_the_text_a_plotting_call_wrote(tmp_path: Path):
    """
    Asserts that a label naming no title and no axis label keeps the title
    and both axis labels a plotting call wrote, setting the tick values in
    the mono face alone.
    """
    chart = Chart(folder=tmp_path, name="kept.png")

    with chart.drawn() as (_, ax):
        ax.set(title="Written", xlabel="Across", ylabel="Down")
        chart.label(ax)

        assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == (
            "Written", "Across", "Down"
        )
        assert ax.get_xticklabels()[0].get_fontfamily() == ["JetBrains Mono"]

