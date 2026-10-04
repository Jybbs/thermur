"""
Pins the records the repository audit reads and reports, meaning the TOML
key each document field reads, the annotation a `Finding` prints, the
finding a `Parity` makes, and the holders a `Manifest` names.
"""

from functools             import reduce
from hypothesis            import example, given
from hypothesis.strategies import text
from pathlib               import Path
from pydantic              import AliasPath, ValidationError
from pytest                import Config, mark, param, raises
from tomllib               import loads
from urllib.parse          import unquote

from thermur.repo.schemas import Author, Config as MiseConfig, Finding, Manifest, Parity


@given(message=text())
@example(message="100%0A")
def test_an_annotation_stays_on_one_line_and_reads_back(message: str):
    """
    Asserts that any message, a newline or a `%` inside it included, writes
    an annotation holding no line break, from which decoding the escapes
    reads the message back, each escape being the percent-encoding of the
    character it stands for.
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
        for document in (Manifest, MiseConfig)
        for name, info in document.model_fields.items()
        if info.is_required()
    ]
)
def test_each_required_field_reads_the_key_its_alias_names(
    document     : type[Manifest | MiseConfig],
    field        : str,
    pytestconfig : Config
):
    """
    Asserts that removing the key a required field's alias names from the
    checkout's own file fails validation at exactly that key, so each field
    reads the TOML key its alias declares.
    """
    alias: AliasPath = document.model_fields[field].validation_alias
    *table, key      = alias.path
    parsed           = loads(
        (pytestconfig.rootpath / document.model_config["title"]).read_text()
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
