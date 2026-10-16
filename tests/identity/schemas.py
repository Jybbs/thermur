"""
Pins what the identity's declaration validates into, meaning each role's
colors read by name, the name a role reads being one the palette declares,
and the faces, the marks, and the tagline as `identity.toml` declares them.
"""

from pydantic import ValidationError
from pytest   import mark, param, raises

from thermur.identity.schemas import Identity, Palette


def test_the_shipped_identity_names_a_file_for_every_face():
    """
    Asserts that every face names at least one file the package ships it as,
    since a chart and a card set their words from those files alone.
    """
    assert all(face.fonts for _, face in Identity.shipped().faces)


def test_the_shipped_identity_orders_the_series_and_the_heat_ramp():
    """
    Asserts that the chart series run from Fire Lord to Viridian and the
    heat ramp from Kindleflame to Burnt Crimson, each in the order a chart
    assigns them.
    """
    palette = Identity.shipped().palette

    assert [pair.light for pair in palette.series] == [
        "fire-lord", "toronto", "topiary-garden", "purpureus", "viridian"
    ]
    assert palette.heat[0] == "kindleflame"
    assert palette.heat[-1] == "burnt-crimson"


def test_the_shipped_identity_reads_each_role_s_colors():
    """
    Asserts that the shipped identity reads the ground, the ink, and the
    lead of each scheme as `visual-identity.md` records them, by the hex
    each color's name files.
    """
    palette = Identity.shipped().palette

    assert [
        palette.hex(name)
        for pair in (palette.ground, palette.ink, palette.lead)
        for name in (pair.light, pair.dark)
    ] == ["#f2f1ee", "#17171b", "#17171b", "#f2f1ee", "#995500", "#c38243"]


@mark.parametrize(
    ("role", "names"),
    [
        param("ink", {"dark": "cararra", "light": "smoke"}, id="pair"),
        param("heat", ["kindleflame", "ember"], id="ramp"),
        param("series", [{"dark": "toronto-lift", "light": "smoke"}], id="series")
    ]
)
def test_a_role_naming_an_undeclared_color_fails(names: object, role: str):
    """
    Asserts that a role naming a color `colors` does not declare fails
    validation, naming the color, whether the role holds a pair, a ramp, or
    a series of pairs.
    """
    palette = Identity.shipped().palette.model_dump() | {role: names}

    with raises(ValidationError, match=r"names undeclared colors: (ember|smoke) "):
        Palette.model_validate(palette)


@mark.parametrize(
    "hex",
    [
        param("#995F00", id="uppercase"),
        param("#995OOO", id="letter-o"),
        param("#99550", id="short"),
        param("#995500ff", id="alpha")
    ]
)
def test_a_color_off_six_lowercase_hex_digits_fails(hex: str):
    """
    Asserts that a color fails validation unless it is `#` and six lowercase
    hex digits.
    """
    palette           = Identity.shipped().palette.model_dump()
    palette["colors"] = palette["colors"] | {"fire-lord-deep": hex}

    with raises(ValidationError):
        Palette.model_validate(palette)


@mark.parametrize("alpha", [0, 1.5])
def test_the_tagline_s_alpha_lies_in_its_range(alpha: float):
    """
    Asserts that the tagline's share of the ink lies above zero and at most
    one.
    """
    identity = Identity.shipped().model_dump()
    identity["tagline"]["alpha"] = alpha

    with raises(ValidationError):
        Identity.model_validate(identity)
