"""
Pins `Piece.declared`, meaning each declaration in a part's folder read
in path order, one outside a part's folder left unread, and a declaration
carrying a field `Piece` lacks refused.
"""

from pathlib  import Path
from pydantic import ValidationError
from pytest   import Config, raises

from thermur.docs.kit import KIT, Piece


def test_a_declaration_carrying_a_field_piece_lacks_is_refused(tmp_path: Path):
    """
    Asserts that a declaration naming a field beyond those a piece declares
    fails validation rather than dropping the field.
    """
    (tmp_path / KIT / "controls").mkdir(parents=True)
    (tmp_path / KIT / "controls/stray.json").write_text(
        '{"examples": [], "exports": ["stray"], "machine": null, "name": "Stray", '
        '"replaces": "A", "when": "B", "extra": "C"}'
    )

    with raises(ValidationError, match="extra"):
        Piece.declared(tmp_path)


def test_each_declaration_in_a_part_s_folder_is_read_in_path_order(tmp_path: Path):
    """
    Asserts that the declarations in two parts' folders are read ordered
    by their paths rather than their names, and that one standing outside a
    part's folder is left unread, as the site's catalog leaves it.
    """
    write(
        examples = [],
        exports  = ["Slider"],
        machine  = "slider",
        name     = "Slider",
        place    = "controls/slider.json",
        replaces = "A",
        root     = tmp_path,
        when     = "B"
    )
    write(
        examples = [{"name": "Ease", "says": "E"}],
        exports  = ["onMotion"],
        machine  = None,
        name     = "The Gate",
        place    = "motion/gate.json",
        replaces = "C",
        root     = tmp_path,
        when     = "D"
    )
    write(
        examples = [],
        exports  = ["stray"],
        machine  = None,
        name     = "Stray",
        place    = "stray.json",
        replaces = "E",
        root     = tmp_path,
        when     = "F"
    )

    slider, gate = Piece.declared(tmp_path)

    assert (slider.name, gate.name) == ("Slider", "The Gate")
    assert gate.machine is None
    assert gate.examples[0].says == "E"


def test_the_shipped_declarations_each_validate(pytestconfig: Config):
    """
    Asserts that every declaration beside the kit's code validates, so The
    Kit lays out a row for each.
    """
    assert Piece.declared(pytestconfig.rootpath)


def write(place: str, root: Path, **fields: object):
    """
    Writes the declaration `fields` set at `place` under the kit's folder in
    the clone at `root`.
    """
    path = root / KIT / place
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(Piece.model_validate(fields).model_dump_json())
