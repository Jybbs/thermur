"""
Holds `Piece`, one piece of the site's widget kit as the declaration beside
its code under `docs/kit/src/` records it, which The Kit lays out as a
table, and `Example`, one example of a piece at work.
"""

from pathlib  import Path
from pydantic import BaseModel
from typing   import Self

KIT = Path("docs/kit/src")
"""
The folder holding the kit's code and each piece's declaration, relative to
the clone's root.
"""


class Example(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One example of a piece at work, which The Kit's catalog runs live.
    """

    name: str
    """
    The example's name, in title case.
    """

    says: str
    """
    What a reader sees the piece do in the example.
    """


class Piece(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One piece of the kit: its name, the names a widget imports it by, the
    Zag.js machine it runs on where one covers it, when a widget reads it,
    what it replaces, and the examples The Kit's catalog runs.
    """

    examples: list[Example]
    """
    The examples of the piece at work.
    """

    exports: list[str]
    """
    The names a widget imports the piece by.
    """

    machine: str | None
    """
    The Zag.js machine the piece runs on, none for a piece no machine
    covers.
    """

    name: str
    """
    The piece's name, in title case.
    """

    replaces: str
    """
    What the piece replaces among the widgets' own code.
    """

    when: str
    """
    When a widget reads the piece.
    """

    @classmethod
    def declared(cls, root: Path) -> list[Self]:
        """
        Validates each declaration in a part's folder under the kit's folder
        in the clone at `root`, in path order, as the site's catalog reads
        them.
        """
        return [
            cls.model_validate_json(path.read_text())
            for path in sorted((root / KIT).glob("*/*.json"))
        ]
