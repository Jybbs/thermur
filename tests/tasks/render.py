"""
Pins what `docs:render` prints and exits with, meaning the path of each file
it rewrote, nothing where every file is current, and each marker naming no
entry in `docs/references.bib`, which fails the task.
"""

from collections.abc import Callable
from common.edits    import append
from pathlib         import Path
from pytest          import CaptureFixture, MonkeyPatch, raises
from types           import ModuleType

from thermur.repo.checkout import Checkout


def test_a_marker_naming_no_entry_fails_the_render(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a page citing a key the bibliography lacks still has every
    other definition written, and that the render then prints the marker to
    standard error and exits `1`.
    """
    monkeypatch.chdir(checkout.root)
    page = Path("docs/pages/notes.md")
    (
        checkout.root / page
    ).write_text("# Notes\n\nA study[^smith2020] beside[^cavagna2010].\n")

    with raises(SystemExit) as exited:
        load_task(".mise/tasks/docs/render.py").main()

    assert exited.value.code
    assert capsys.readouterr() == (
        f"{page}\n",
        f"Line 3 of `{page}` cites `smith2020`, which names no entry in "
        "`docs/references.bib`\n"
    )
    assert (
        checkout.read(page)
                .endswith(f"{checkout.bibliography.cite('cavagna2010')}\n")
    )


def test_the_render_prints_each_file_it_rewrote_and_then_nothing(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a first render over a copy missing the identity and holding
    a page that cites a study it does not define prints the path of each
    file it wrote and exits `0`, and a second render prints nothing.
    """
    monkeypatch.chdir(checkout.root)
    (checkout.root / "docs/.vitepress/lib/rendered/identity.json").unlink()
    append(checkout, "docs/pages/index.md", "\nOrdered flocks[^cavagna2010].\n")

    for printed in (
        "docs/.vitepress/lib/rendered/identity.json\ndocs/pages/index.md\n", ""
    ):
        with raises(SystemExit) as exited:
            load_task(".mise/tasks/docs/render.py").main()

        assert not exited.value.code
        assert capsys.readouterr().out == printed
