"""
Pins what the render task writes and how it reads the copies the repository
holds, meaning the identity as JSON, each page carrying the definition of
every study it cites, a drifted or missing copy reported, only a drifted
copy written again, and each marker naming no entry listed on its page.
"""

from common.edits import append
from json         import loads
from pathlib      import Path

from thermur.docs.outputs     import Outputs
from thermur.identity.schemas import Identity
from thermur.repo.checkout    import Checkout
from thermur.repo.schemas     import Finding


def test_a_missing_or_edited_copy_drifts_and_is_written_again(checkout: Checkout):
    """
    Asserts that a missing identity and a page citing a study it does not
    define both count as drifted, that writing rewrites each, and that a
    second write finds nothing to write.
    """
    outputs = Outputs(checkout=checkout)
    page    = Path("docs/pages/index.md")
    (checkout.root / outputs.identity).unlink()
    append(checkout, str(page), "\nTurns travel[^attanasi2014].\n")

    assert outputs.drifted == [outputs.identity, page]
    assert outputs.write() == [outputs.identity, page]
    assert outputs.write() == []
    assert (
        checkout.read(page)
                .endswith(f"\n\n{checkout.bibliography.cite('attanasi2014')}\n")
    )


def test_a_page_git_ignores_is_left_alone(checkout: Checkout):
    """
    Asserts that a Markdown file under a folder `.gitignore` names, such
    as a package's readme under `docs/node_modules/`, is neither read nor
    written, however it cites.
    """
    ignored = checkout.root / "docs/node_modules/package/README.md"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("A study[^smith2020] and another[^cavagna2010].\n")

    outputs = Outputs(checkout=checkout)

    assert (outputs.drifted, outputs.unknown) == ([], [])


def test_each_marker_naming_no_entry_is_listed_on_its_page(checkout: Checkout):
    """
    Asserts that a marker under a key the bibliography lacks is listed with
    its page and its line, on a page git would track though nothing has
    added it yet, and that the page's other citations still drift until
    written.
    """
    page = Path("docs/pages/notes.md")
    (
        checkout.root / page
    ).write_text("# Notes\n\nA study[^smith2020] beside[^cavagna2010].\n")
    outputs = Outputs(checkout=checkout)

    assert outputs.unknown == [
        Finding(
            file    = page,
            message = f"Line 3 of `{page}` cites `smith2020`, which names no entry in "
            "`docs/references.bib`"
        )
    ]
    assert outputs.drifted == [page]


def test_the_render_writes_the_identity_as_json():
    """
    Asserts that the identity file holds the identity as `Identity.shipped`
    reads it, so the site reads the very records the chart theme does.
    """
    outputs = Outputs()

    assert loads(outputs.written[outputs.identity]) == Identity.shipped().model_dump(
        mode = "json"
    )
