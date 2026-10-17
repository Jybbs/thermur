"""
Pins how a Markdown page's footnotes are found and rewritten, meaning the
definition of each study a page cites written at its end in the order the
page first cites it, a stale one removed, a footnote under any other label
kept, and each marker naming no entry reported at its line.
"""

from pytest  import Config, fixture, mark, param

from thermur.docs.pages   import Page
from thermur.docs.schemas import Bibliography


@fixture(scope="module")
def shipped(pytestconfig: Config) -> Bibliography:
    """
    Reads the bibliography the repository ships.
    """
    return Bibliography.read(pytestconfig.rootpath)


def test_a_definition_is_rewritten_and_a_stale_one_removed(shipped: Bibliography):
    """
    Asserts that a definition differing from its entry is written again at
    the end, and that the definition of a study the page no longer cites is
    removed with the blank line the parser reads as part of it.
    """
    page = Page(
        text = "Ordered flocks[^cavagna2010].\n\n[^higham2001]: Higham 2001.\n\n"
        "Closing paragraph.\n\n[^cavagna2010]: Cavagna et al. 2010, by hand.\n"
    )

    assert page.cited(shipped) == (
        "Ordered flocks[^cavagna2010].\n\nClosing paragraph.\n\n"
        f"{shipped.cite('cavagna2010')}\n"
    )


def test_a_footnote_under_another_label_is_kept(shipped: Bibliography):
    """
    Asserts that a numbered footnote, and a marker and definition under a
    key the bibliography lacks, stay where the page writes them.
    """
    page = Page(
        text = "A note[^1] and a study[^smith2020] beside[^cavagna2010].\n\n"
        "[^1]: A note.\n\n[^smith2020]: Smith 2020, by hand.\n"
    )

    assert page.cited(
        shipped
    ) == f"{page.text.rstrip()}\n\n{shipped.cite('cavagna2010')}\n"


def test_a_marker_naming_no_entry_is_listed_at_its_line(shipped: Bibliography):
    """
    Asserts that a marker under a study's key the bibliography lacks is
    listed at the first line of the block it sits in, a table row's line
    included, whereas a numbered marker, one naming an entry, and one
    carrying two letters after its year are not.
    """
    page = Page(
        text = "# Title\n\nA note[^1] and a study[^cavagna2010].\n\n"
        "A paragraph\nwhose second line cites[^smith2020] and[^smith2020ab].\n\n"
        "| **Norm** | **Source** |\n|---|---|\n| Polarization | [^jones2019a] |\n"
    )

    assert page.unknown(shipped) == [(5, "smith2020"), (10, "jones2019a")]


@mark.parametrize(
    "text",
    [
        param("A plain page.", id="no-footnote"),
        param("A note[^1].\n\n[^1]: A note.\n", id="numbered"),
        param("The form `[^cavagna2010]` in a code span.\n", id="code-span"),
        param("```\n[^cavagna2010]\n```\n", id="fence"),
        param("<div>[^cavagna2010]</div>\n", id="html-block")
    ]
)
def test_a_page_citing_no_study_is_left_as_it_stands(shipped: Bibliography, text: str):
    """
    Asserts that a page whose only marker sits in a code span, a fence, or
    a block of HTML, which VitePress renders as written, or which carries no
    footnote of a study, is returned byte for byte, with no closing newline
    added.
    """
    assert Page(text=text).cited(shipped) == text


def test_a_page_citing_no_study_loses_a_stale_definition(shipped: Bibliography):
    """
    Asserts that a page holding the definition of a study it no longer cites
    loses that definition, gains none, and ends on one newline.
    """
    page = Page(
        text = "# Notes\n\nA paragraph.\n\n[^cavagna2010]: Cavagna 2010, stale.\n"
    )

    assert page.cited(shipped) == "# Notes\n\nA paragraph.\n"


def test_each_cited_study_is_defined_at_the_end_in_first_cited_order(
    shipped: Bibliography
):
    """
    Asserts that a page citing two studies, one of them twice and out of
    their keys' order, gains one definition of each at its end, in the order
    the page first cites them, with a blank line before them and between
    them.
    """
    page = Page(
        text = "# Turning\n\nInertia[^cavagna2015] sets the speed a turn "
        "travels at[^attanasi2014].\n\nIt fades without it[^cavagna2015].\n"
    )

    assert page.cited(shipped) == (
        f"{page.text.rstrip()}\n\n{shipped.cite('cavagna2015', 'attanasi2014')}\n"
    )


def test_rewriting_a_rewritten_page_changes_nothing(shipped: Bibliography):
    """
    Asserts that the render reads its own output as current, so a page it
    wrote once is never rewritten.
    """
    once = Page(text="Waves[^storms2019] travel[^procaccini2011].").cited(shipped)

    assert Page(text=once).cited(shipped) == once
