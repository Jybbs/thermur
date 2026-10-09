"""
Pins the records the repository audit reads and reports, meaning the value
a document's text reads as and the key each document field reads, the
annotation a `Finding` prints, the finding a `Parity` makes, the holders
a `Manifest` names, the labels a `Template` lists, and the rows a `Labels`
registry accepts, the command each `Label` writes, and the live labels a
registry omits.
"""

from functools    import reduce
from pathlib      import Path
from pydantic     import AliasPath, ValidationError
from pytest       import Config, mark, param, raises
from urllib.parse import unquote

from thermur.repo.schemas import (
    Author, Document, Finding, Label, Labels, Manifest, Parity, Template
)

ROW = {"color": "8c055e", "description": "Wrong output", "name": "🐞 bug"}


@mark.parametrize(
    "message",
    [
        param("line\nbreak", id="newline"),
        param("carriage\rreturn", id="carriage-return"),
        param("100%", id="percent"),
        param("a,b:c", id="separators"),
        param("100%0A", id="escape-literal")
    ]
)
def test_an_annotation_stays_on_one_line_and_reads_back(message: str):
    """
    Asserts that a message holding a line break, a `%`, a separator, or the
    literal text of an escape writes an annotation holding no line break,
    from which decoding the escapes reads the message back.
    """
    annotation = Finding(file=Path("pyproject.toml"), message=message).annotation

    assert not {"\n", "\r"} & set(annotation)
    assert unquote(annotation.removeprefix("::error file=pyproject.toml::")) == message


@mark.parametrize(
    ("file", "annotation"),
    [
        param(".mise/bin/prose", "::error file=.mise/bin/prose::m", id="plain"),
        param("a,b:c", "::error file=a%2Cb%3Ac::m", id="separators")
    ]
)
def test_an_annotation_names_its_file_for_the_runner(file: str, annotation: str):
    """
    Asserts that a finding prints as the error annotation GitHub Actions
    places on the file it names, escaping a `,` or a `:` in that name, since
    a workflow command reads either as the end of the property.
    """
    assert Finding(file=Path(file), message="m").annotation == annotation


@mark.parametrize(
    ("copied", "finding"),
    [
        param("3.14", None, id="agrees"),
        param(
            None,
            Finding(
                file    = Path("README.md"),
                message = "The badge is missing, where the manifest reads `3.14`"
            ),
            id = "missing"
        ),
        param(
            "3.13",
            Finding(
                file    = Path("README.md"),
                message = "The badge reads `3.13`, where the manifest reads `3.14`"
            ),
            id = "drifts"
        )
    ]
)
def test_a_parity_names_a_copy_that_drifts(copied: str | None, finding: Finding | None):
    """
    Asserts that a restated declaration reading another value, or missing
    from its file, makes a finding on that file naming both values, and that
    one reading the original value makes none.
    """
    parity = Parity(
        copied   = copied,
        file     = Path("README.md"),
        label    = "The badge",
        origin   = "the manifest",
        original = "3.14"
    )

    assert parity.finding == finding


@mark.parametrize(
    ("document", "field"),
    [
        param(document, name, id=f"{document.__name__}.{name}")
        for document in Document.__subclasses__()
        if "title" in document.model_config
        for name, info in document.model_fields.items()
        if info.is_required()
    ]
)
def test_each_required_field_reads_the_key_its_alias_names(
    document     : type[Document],
    field        : str,
    pytestconfig : Config
):
    """
    Asserts that removing the key a required field's alias names from the
    checkout's own file fails validation at exactly that key, so each field
    reads the TOML or YAML key its alias declares.
    """
    alias: AliasPath = document.model_fields[field].validation_alias
    *table, key      = alias.path
    file             = Path(document.model_config["title"])
    parsed           = document.load(
        file.suffix,
        (pytestconfig.rootpath / file).read_text(encoding="utf-8")
    )
    del reduce(dict.__getitem__, table, parsed)[key]

    with raises(ValidationError) as error:
        document.model_validate(parsed)

    assert [detail["loc"] for detail in error.value.errors()] == [tuple(alias.path)]


@mark.parametrize(
    ("names", "holders"),
    [
        param(["Ada"], "Ada", id="one"),
        param(["Ada", "Bea"], "Ada and Bea", id="two"),
        param(["Ada", "Bea", "Cy"], "Ada, Bea, and Cy", id="three")
    ]
)
def test_the_authors_join_the_way_a_copyright_line_names_them(
    names   : list[str],
    holders : str
):
    """
    Asserts that one author reads alone, two join by `and`, and three or
    more join by commas with `and` before the last.
    """
    manifest = Manifest.model_construct(authors=[Author(name=name) for name in names])

    assert manifest.holders == holders


@mark.parametrize(
    ("data", "errors"),
    [
        param(
            {"labels": [ROW | {"color": "8C055E"}]},
            [(("labels", 0, "color"), "String should match pattern '^[0-9a-f]{6}$'")],
            id = "uppercase-color"
        ),
        param(
            {"labels": [ROW | {"color": "#8c055e"}]},
            [(("labels", 0, "color"), "String should match pattern '^[0-9a-f]{6}$'")],
            id = "hashed-color"
        ),
        param(
            {"labels": [ROW | {"color": "fff"}]},
            [(("labels", 0, "color"), "String should match pattern '^[0-9a-f]{6}$'")],
            id = "short-color"
        ),
        param(
            {"labels": [ROW | {"glyph": "🐞"}]},
            [(("labels", 0, "glyph"), "Extra inputs are not permitted")],
            id = "further-key"
        ),
        param(
            {"labels": [{"color": "8c055e", "name": "🐞 bug"}]},
            [(("labels", 0, "description"), "Field required")],
            id = "missing-description"
        ),
        param(
            {"categories": [], "labels": [ROW]},
            [(("categories",), "Extra inputs are not permitted")],
            id = "key-beside-the-rows"
        ),
        param(
            {"labels": [ROW, ROW | {"color": "0044aa"}]},
            [(("labels",), "Value error, declares the name `🐞 bug` more than once")],
            id = "repeated-name"
        ),
        param(
            {"labels": [ROW, ROW | {"name": "🧰 tooling"}]},
            [(("labels",), "Value error, declares the color `8c055e` more than once")],
            id = "repeated-color"
        ),
        param(
            {"labels": [ROW, ROW]},
            [
                (
                    ("labels",),
                    "Value error, declares the name `🐞 bug` and the color `8c055e` "
                    "more than once"
                )
            ],
            id = "repeated-row"
        )
    ]
)
def test_a_malformed_registry_fails_at_the_key_it_breaks(
    data   : dict[str, list[dict[str, str]]],
    errors : list[tuple[tuple[int | str, ...], str]]
):
    """
    Asserts that a registry holding a color other than six lowercase hex
    digits, a row with a further key or a missing one, a key beside its
    rows, or two labels sharing a name or a color fails validation at the
    key that breaks the rule, with the message a finding then carries.
    """
    with raises(ValidationError) as error:
        Labels.model_validate(data)

    assert [(detail["loc"], detail["msg"]) for detail in error.value.errors()] == errors


def test_a_label_description_runs_to_100_characters():
    """
    Asserts that a description of 100 characters, the most GitHub's label
    API accepts, validates, whereas one of 101 fails at the description.
    """
    assert len(
        Label.model_validate(ROW | {"description": "x" * 100}).description
    ) == 100

    with raises(ValidationError) as error:
        Label.model_validate(ROW | {"description": "x" * 101})

    assert [(detail["loc"], detail["msg"]) for detail in error.value.errors()] == [
        (("description",), "String should have at most 100 characters")
    ]


@mark.parametrize(
    ("live", "omitted"),
    [
        param(["🐞 bug"], [], id="all-declared"),
        param(
            ["✨feature", "🐞 bug", "🐞bug"],
            ["✨feature", "🐞bug"],
            id = "older-labels"
        ),
        param([], [], id="none-live")
    ]
)
def test_a_registry_omits_each_live_label_it_declares_nowhere(
    live    : list[str],
    omitted : list[str]
):
    """
    Asserts that a registry names each live label it declares no row for, in
    the order the live labels arrive, and none where it declares them all.
    """
    assert Labels.model_validate({"labels": [ROW]}).omits(live) == omitted


@mark.parametrize(
    ("suffix", "text", "value"),
    [
        param(".toml", 'version = "0.1.0"\n', {"version": "0.1.0"}, id="toml"),
        param(
            ".yml",
            "on: push\nversion: 0.1.0\n",
            {"on": "push", "version": "0.1.0"},
            id = "yaml"
        ),
        param(".cff", "version: '0.1.0'\n", {"version": "0.1.0"}, id="cff"),
        param(
            ".md",
            "---\nname: Bug\nlabels: []\n---\n\n---\n\nname: Body\n",
            {"name": "Bug", "labels": []},
            id = "front-matter"
        ),
        param(
            ".cff",
            "date-released: 2026-10-02\n",
            {"date-released": "2026-10-02"},
            id = "timestamp"
        )
    ]
)
def test_a_document_reads_the_value_its_suffix_names(
    suffix : str,
    text   : str,
    value  : object
):
    """
    Asserts that a document's text reads as TOML under `.toml`, as YAML
    1.2 under any other suffix, keeping `on`, a version, and a date as the
    strings they are written as, and as the front matter alone under `.md`,
    ending at its first closing `---`.
    """
    assert Document.load(suffix, text) == value


def test_a_label_writes_the_command_that_creates_or_updates_it():
    """
    Asserts that a label writes the `gh label create --force` command
    carrying its color, its description, and its name, which creates the
    label or updates the one already carrying that name.
    """
    assert Label.model_validate(ROW).command == [
        "gh", "label", "create", "--color", "8c055e",
        "--description", "Wrong output", "--force", "🐞 bug"
    ]


@mark.parametrize(
    ("labels", "names"),
    [
        param(["🐞 bug", "🦜 cli"], ["🐞 bug", "🦜 cli"], id="list"),
        param("🐞 bug, 🦜 cli", ["🐞 bug", "🦜 cli"], id="comma-delimited"),
        param("🐞 bug", ["🐞 bug"], id="one"),
        param("", [], id="empty")
    ]
)
def test_a_template_reads_its_labels_from_a_list_or_one_string(
    labels : str | list[str],
    names  : list[str]
):
    """
    Asserts that a template's labels read from a list as written, and from
    one string as each name its commas separate with the spaces around it
    removed, so an empty string names none.
    """
    assert Template.model_validate({"labels": labels}).labels == names


def test_a_markdown_file_opening_on_no_front_matter_fails_to_read():
    """
    Asserts that a Markdown file whose text opens on anything but front
    matter fails to read rather than reading as an empty document.
    """
    with raises(ValueError, match="opens on no front matter"):
        Document.load(".md", "name: Bug\n---\n")
