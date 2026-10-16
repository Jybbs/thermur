"""
Pins the rcParams the chart theme builds from the identity, and that it
registers each face the package ships with matplotlib, so a chart sets its
words in the identity's faces whatever the machine has installed.
"""

from dataclasses         import replace
from importlib.resources import files
from matplotlib          import font_manager
from pathlib             import Path
from pytest              import MonkeyPatch, mark

from thermur.figures.theme    import Theme
from thermur.identity.schemas import Faces, Identity


def test_each_size_steps_up_from_the_ticks_to_the_title():
    """
    Asserts that the tick values, the axis labels, and the title each take a
    larger size than the last, the axis labels setting the default size.
    """
    params = Theme().params
    label  = params["axes.labelsize"]

    assert params["xtick.labelsize"] < label < params["axes.titlesize"]
    assert params["font.size"] == label


def test_the_theme_assigns_the_series_in_the_identity_s_order():
    """
    Asserts that the color cycle runs through the light scheme's series in
    the order the identity declares them.
    """
    assert Theme().params["axes.prop_cycle"].by_key()["color"] == [
        "#bb7733", "#5069ad", "#6f7c00", "#9a4eae", "#1e9167"
    ]


def test_the_theme_colors_every_surface_in_the_light_scheme():
    """
    Asserts that the figure and every axes take the light scheme's ground,
    and the text, the axes, and the ticks its ink.
    """
    params = Theme().params

    assert {params[key] for key in (
        "axes.facecolor", "figure.facecolor"
    )} == {"#f2f1ee"}
    assert {
        params[key]
        for key in ("axes.edgecolor", "axes.labelcolor", "text.color", "xtick.color")
    } == {"#17171b"}


def test_the_theme_reads_the_identity_it_is_given():
    """
    Asserts that a theme built over an identity whose light ground differs
    colors its surfaces in that ground, so nothing in the theme restates
    a color.
    """
    identity = Identity.shipped()
    palette  = identity.palette.model_copy(
        update = {"colors": identity.palette.colors | {"cararra": "#ffffff"}}
    )

    theme = Theme(identity=identity.model_copy(update={"palette": palette}))

    assert theme.params["figure.facecolor"] == "#ffffff"


def test_the_theme_registers_each_face_the_package_ships():
    """
    Asserts that building the theme sets the body face as the default
    family, and that a second theme registers the faces again in place of
    the first's rather than beside them.
    """
    assert Theme().params["font.family"] == ["Crimson Pro"]

    known = len(font_manager.fontManager.ttflist)
    Theme().families
    assert len(font_manager.fontManager.ttflist) == known


@mark.parametrize("kind", Faces.model_fields)
def test_each_face_the_package_ships_beats_a_copy_the_machine_has(
    kind        : str,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the face the theme registers for each kind of text is the
    one matplotlib finds, even where a copy of the same family the machine
    has installed stands ahead of it.
    """
    family  = Theme().families[kind]
    manager = font_manager.fontManager
    shipped = next(entry for entry in manager.ttflist if entry.name == family)
    monkeypatch.setattr(
        manager,
        "ttflist",
        [replace(shipped, fname="/machine/copy.ttf"), *manager.ttflist]
    )

    Theme().families
    assert "/machine/copy.ttf" not in {entry.fname for entry in manager.ttflist}
    assert Path(
        font_manager.findfont(family, fallback_to_default=False)
    ).is_relative_to(
        Path(str(files("thermur.figures") / "fonts"))
    )
