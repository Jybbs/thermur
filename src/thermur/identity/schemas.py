"""
Holds the records `identity.toml` beside this module validates into, meaning
the `Identity` read through `Identity.shipped()`, its `Palette` of named
colors with the `Pair` each role takes per color scheme, its `Faces` with
each `Face` and the `Font` files that face ships as, its `Marks` naming the
logo's files, and its `Tagline`.
"""

from importlib.resources import read_text
from pathlib             import Path
from pydantic            import BaseModel, Field, model_validator
from tomllib             import loads
from typing              import Annotated, Literal, Self

type Hex = Annotated[str, Field(pattern=r"^#[0-9a-f]{6}$")]


class Font(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One file a face ships as, under the folder `thermur.figures` keeps the
    faces in.
    """

    file: Path
    """
    The file's path under that folder.
    """

    style: Literal["italic", "normal"]
    """
    The style the file sets, as CSS names it.
    """

    weight: int
    """
    The weight the file sets, as CSS numbers it.
    """


class Pair(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The one value a role takes under each color scheme.
    """

    dark: str
    """
    The value under the dark scheme.
    """

    light: str
    """
    The value under the light scheme.
    """


class Tagline(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The line the identity sets beneath the wordmark, in the body face's
    italic and the theme's ink laid over the ground.
    """

    alpha: Annotated[float, Field(gt=0, le=1)]
    """
    The share of the ink laid over the ground the tagline is set in.
    """

    text: str
    """
    The tagline's words.
    """


class Face(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One face the identity sets a kind of text in, beside the faces a reader
    falls back on where it does not load.
    """

    fallbacks: tuple[str, ...]
    """
    The faces a reader falls back on, in order, ending on a generic family.
    """

    family: str
    """
    The face's family name, as Google Fonts and each font file name it.
    """

    fonts: tuple[Font, ...]
    """
    The files the package ships the face as.
    """

    weights: tuple[int, ...]
    """
    The weights the site loads the face at.
    """

    italic: bool = False
    """
    Whether the site loads the face's italic beside its roman.
    """


class Marks(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The files the logo ships as, each drawn once for a light page and once
    for a dark one.
    """

    mark: Pair
    """
    The mark, every bird of the original logo along the flock's gradient.
    """

    small: Pair
    """
    The mark drawn below 40 pixels, keeping only the original's larger
    birds, for a favicon or a badge.
    """

    wordmark: Pair
    """
    The wordmark, the name set in the display face breaking into birds.
    """


class Palette(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    Every color the identity names, beside the color each role takes, each
    role naming its color by the key `colors` files it under.
    """

    colors: dict[str, Hex]
    """
    Each color's hex, by its name in lowercase with hyphens between its
    words.
    """

    critical: Pair
    """
    The color of a failed state, always beside an icon and a label.
    """

    ground: Pair
    """
    The color of the page and every chart surface.
    """

    heat: tuple[str, ...]
    """
    The ramp coloring temperature alone, from light to dark.
    """

    ink: Pair
    """
    The color of text, axes, and every label on a chart.
    """

    lead: Pair
    """
    The site's accent and the color of its links.
    """

    series: tuple[Pair, ...]
    """
    The categorical series of a chart, in the order a chart assigns them.
    """

    warning: Pair
    """
    The color of a state worth attention, always beside an icon and a
    label.
    """

    @property
    def named(self) -> set[str]:
        """
        Collects every color name a role reads.
        """
        pairs = [role for _, role in self if isinstance(role, Pair)] + list(self.series)
        return {
            *self.heat,
            *(name for pair in pairs for name in (pair.dark, pair.light))
        }

    def hex(self, name: str) -> str:
        """
        Reads the hex of the color filed under `name`.
        """
        return self.colors[name]

    @model_validator(mode="after")
    def reject_unknown(self) -> Self:
        """
        Passes the palette through where every role names a color `colors`
        declares.

        Raises:
            ValueError: Naming each color a role reads that `colors` does
                        not declare, which validation reports against the
                        palette.
        """
        if unknown := sorted(self.named - self.colors.keys()):
            raise ValueError(f"names undeclared colors: {', '.join(unknown)}")

        return self


class Faces(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The face each kind of text takes.
    """

    body: Face
    """
    The face of running text and every caption.
    """

    display: Face
    """
    The face of the wordmark, page titles, and headings.
    """

    mono: Face
    """
    The face of code, file paths, readings, and numbers that line up in
    columns.
    """


class Identity(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The palette, the faces, the marks, and the tagline the site's theme and
    the chart theme both read.
    """

    faces: Faces
    """
    The face each kind of text takes.
    """

    marks: Marks
    """
    The files the logo ships as.
    """

    palette: Palette
    """
    Every color the identity names and the color each role takes.
    """

    tagline: Tagline
    """
    The line beneath the wordmark.
    """

    @classmethod
    def shipped(cls) -> Self:
        """
        Validates `identity.toml`, the declaration beside this module.
        """
        return cls.model_validate(
            loads(read_text(__package__, "identity.toml"))
        )
