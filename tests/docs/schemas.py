"""
Pins what `docs/references.bib` reads into, meaning the footnote definition
each kind of study takes, the key each study's first author and year hold
it to, the fields an entry may not leave out, and the bibliography the
repository ships.
"""

from collections.abc import Callable
from pathlib         import Path
from pydantic        import ValidationError
from pytest          import Config, fixture, mark, param, raises
from re              import fullmatch, sub

from thermur.docs.schemas import Article, Bibliography, Preprint, Study

ARTICLE = """@article{{{key},
    author   = {{{author}}},
    title    = {{{title}}},
    journal  = {{Bioinspiration \\& Biomimetics}},
    year     = {{2014}},
    volume   = {{9}},{number}
    pages    = {{1--10}},
    doi      = {{10.1088/1748-3182/9/2/025012}},
    priority = {{prio2}},
    annote   = {{What the study measured.}}
}}
"""


@fixture
def read(tmp_path: Path) -> Callable[..., Bibliography]:
    """
    Reads a bibliography holding one `@article` entry, its fields filled
    from the keywords given over a study of 2014 by Virágh and colleagues.

    Returns:
        A reader taking any field of `ARTICLE` as a keyword, or `text` for
        the whole file.
    """
    def entry(text: str | None = None, **fields: str) -> Bibliography:
        """
        Writes the file under `tmp_path` and validates it.
        """
        (file := tmp_path / "docs/references.bib").parent.mkdir(exist_ok=True)
        file.write_text(
            text or ARTICLE.format(
                **{
                    "author" : "Virágh, Csaba and Vásárhelyi, Gábor and Vicsek, Tamás",
                    "key"    : "viragh2014",
                    "number" : "\n    number   = {2},",
                    "title"  : "Flocking algorithm for autonomous flying robots"
                }
                | fields
            ),
            encoding = "utf-8"
        )
        return Bibliography.read(tmp_path)

    return entry


@mark.parametrize(
    ("author", "key", "named"),
    [
        param("Higham, Desmond J.", "higham2014", "Higham", id="one"),
        param(
            "Moisseeva, Nadya and Stull, Roland",
            "moisseeva2014",
            "Moisseeva and Stull",
            id = "two"
        ),
        param(
            "Metz, Luke and Freeman, C. Daniel and Kachman, Tal",
            "metz2014",
            "Metz et al.",
            id = "three"
        ),
        param(
            "van Zuidam, Lars and Hildenbrandt, Hanno",
            "vanzuidam2014",
            "van Zuidam and Hildenbrandt",
            id = "particle"
        )
    ]
)
def test_a_definition_names_the_authors_by_their_count(
    author : str,
    key    : str,
    named  : str,
    read   : Callable[..., Bibliography]
):
    """
    Asserts that a definition names one author alone, both of two, and the
    first of three or more followed by `et al.`, each surname keeping the
    particle the entry gives it.
    """
    assert read(author=author, key=key).studies[key].authors == named


def test_a_block_the_parser_cannot_read_is_named_by_its_line(
    read: Callable[..., Bibliography]
):
    """
    Asserts that a second entry under a key the file already holds fails
    validation naming the line its block opens on, rather than replacing the
    first entry.
    """
    entry = ARTICLE.format(
        author = "Virágh, Csaba",
        key    = "viragh2014",
        number = "",
        title  = "X"
    )

    with raises(
        ValidationError,
        match = "line 13 holds a block bibtexparser cannot read"
    ):
        read(text=f"{entry}\n{entry}")


def test_a_preprint_names_its_archive_and_identifier(read: Callable[..., Bibliography]):
    """
    Asserts that a `@misc` entry naming an archive reads as a preprint
    there, its identifier in place of a journal's volume and pages.
    """
    study = read(
        text = """@misc{metz2021,
    author        = {Metz, Luke and Freeman, C. Daniel and Kachman, Tal},
    title         = {Gradients are Not All You Need},
    year          = {2021},
    archiveprefix = {arXiv},
    eprint        = {2111.05803},
    doi           = {10.48550/arXiv.2111.05803},
    priority      = {prio3},
    annote        = {What the report argues.}
}
"""
    ).studies["metz2021"]

    assert isinstance(study, Preprint)
    assert study.definition == (
        '[^metz2021]: Metz et al. 2021. "Gradients are Not All You Need." '
        "arXiv preprint 2111.05803. <https://doi.org/10.48550/arXiv.2111.05803>"
    )


@mark.parametrize("mark", ["?", "!"])
def test_a_title_ending_on_its_own_mark_takes_no_period(
    mark : str,
    read : Callable[..., Bibliography]
):
    """
    Asserts that a title closing on a question mark or an exclamation mark
    keeps that mark alone inside its quotes.
    """
    study = read(title=f"Do drones flock{mark}").studies["viragh2014"]

    assert f'"Do drones flock{mark}" *' in study.definition


def test_an_article_reads_in_the_footnote_form(read: Callable[..., Bibliography]):
    """
    Asserts that an article's definition names its authors, year, quoted
    title, journal in italics, volume, issue, and pages, then its DOI's
    address, with the BibTeX dash, ampersand, and italics decoded into the
    marks a footnote shows.
    """
    study = read(
        title = "Flocking of \\textit{Sturnus vulgaris} and drones"
    ).studies["viragh2014"]

    assert isinstance(study, Article)
    assert study.definition == (
        '[^viragh2014]: Virágh et al. 2014. "Flocking of *Sturnus vulgaris* and '
        'drones." *Bioinspiration & Biomimetics* 9(2): 1–10. '
        "<https://doi.org/10.1088/1748-3182/9/2/025012>"
    )


@mark.parametrize(
    ("author", "key"),
    [
        param(
            "D'Orsogna, Maria R. and Chuang, Yao-Li",
            "dorsogna2014",
            id = "apostrophe"
        ),
        param("Vásárhelyi, Gábor and Virágh, Csaba", "vasarhelyi2014", id="accents"),
        param("Virágh, Csaba and Vásárhelyi, Gábor", "viragh2014b", id="letter")
    ]
)
def test_a_key_folds_the_first_surname_to_lowercase_letters(
    author : str,
    key    : str,
    read   : Callable[..., Bibliography]
):
    """
    Asserts that a key spells the first author's surname in lowercase ASCII
    letters alone before the year, dropping accents and apostrophes, and
    takes one letter after the year.
    """
    assert key in read(author=author, key=key).studies


@mark.parametrize(
    "key",
    [
        param("vasarhelyi2014", id="second-author"),
        param("viragh2015", id="year"),
        param("viragh2014ab", id="two-letters")
    ]
)
def test_a_key_its_first_author_and_year_do_not_give_is_refused(
    key  : str,
    read : Callable[..., Bibliography]
):
    """
    Asserts that a key opening on anything but the first author's folded
    surname and the year, or carrying two letters after it, fails validation
    naming the key the entry gives.
    """
    with raises(
        ValidationError,
        match = "where its first author and year give `viragh2014`"
    ):
        read(key=key)


@mark.parametrize(
    ("old", "new"),
    [
        param("    annote   = {What the study measured.}\n", "", id="annote"),
        param("    author   = {Virágh, Csaba},\n", "", id="author"),
        param("    priority = {prio2},\n", "", id="priority"),
        param("{prio2}", "{prio4}", id="unknown-priority"),
        param("@article", "@book", id="unknown-type"),
        param("    pages    = {1--10},\n", "", id="pages")
    ]
)
def test_an_entry_leaving_out_a_field_its_kind_reads_is_refused(
    new  : str,
    old  : str,
    read : Callable[..., Bibliography]
):
    """
    Asserts that an entry without an annotation, an author, a priority, or
    the pages an article needs fails validation, as does a priority JabRef
    does not write and an entry type no study takes.
    """
    text = ARTICLE.format(
        author = "Virágh, Csaba",
        key    = "viragh2014",
        number = "",
        title  = "Flocking"
    )

    with raises(ValidationError):
        read(text=text.replace(old, new))


def test_an_article_with_no_issue_leaves_the_parentheses_out(
    read: Callable[..., Bibliography]
):
    """
    Asserts that a journal numbering no issue within the volume reads as the
    volume followed directly by the pages.
    """
    study = read(number="").studies["viragh2014"]

    assert study.venue == "*Bioinspiration & Biomimetics* 9: 1–10"


def test_capitalized_keys_and_latex_accents_read_the_same(
    read: Callable[..., Bibliography]
):
    """
    Asserts that an entry written with the capitalized type JabRef writes,
    capitalized field keys, and LaTeX accents reads into the definition the
    lowercase form gives.
    """
    text = ARTICLE.format(
        author = "Vir{\\'a}gh, Csaba",
        key    = "viragh2014",
        number = "",
        title  = "X"
    )
    jabref = sub(
        r"(?m)^    (\w+)",
        lambda field: f"    {field[1].title()}",
        text.replace("@article", "@Article")
    )
    plain = read(text=text.replace("Vir{\\'a}gh", "Virágh")).studies["viragh2014"]

    assert read(text=jabref).studies["viragh2014"].definition == plain.definition


def test_citing_keys_joins_their_definitions_in_the_order_given(
    read: Callable[..., Bibliography]
):
    """
    Asserts that citing several keys writes each definition in the order the
    keys are given, with a blank line between two of them, and that each key
    no entry holds is named in the order given.
    """
    bibliography = read(
        text = "\n".join(
            ARTICLE.format(
                author = author,
                key    = key,
                number = "",
                title  = key
            )
            for author, key in (
                ("Virágh, Csaba", "viragh2014"),
                ("Vicsek, Tamás", "vicsek2014")
            )
        )
    )
    studies = bibliography.studies

    assert bibliography.cite("vicsek2014", "viragh2014") == (
        f"{studies['vicsek2014'].definition}\n\n{studies['viragh2014'].definition}"
    )
    assert bibliography.unknown("smith2020", "viragh2014", "jones2019") == [
        "`smith2020` names no entry in `docs/references.bib`",
        "`jones2019` names no entry in `docs/references.bib`"
    ]


def test_the_shipped_bibliography_lists_each_study_in_key_order(pytestconfig: Config):
    """
    Asserts that every entry in the repository's `docs/references.bib`
    validates, sits in the order of its key, carries a key in the form
    `Study.form` sets, and reads as a definition holding no run of spaces,
    the mark of a LaTeX special left unescaped.
    """
    studies = Bibliography.read(pytestconfig.rootpath).studies

    assert list(studies) == sorted(studies)
    for key, study in studies.items():
        assert fullmatch(Study.form, key)
        assert "  " not in study.definition
        assert "\xa0" not in study.annote + study.definition
