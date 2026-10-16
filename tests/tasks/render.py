"""
Pins what `docs:render` prints, meaning the path of each file it rewrote and
nothing where every file is current.
"""

from collections.abc import Callable
from pathlib         import Path
from pytest          import CaptureFixture, MonkeyPatch
from types           import ModuleType


def test_the_render_prints_each_file_it_rewrote_and_then_nothing(
    capsys      : CaptureFixture[str],
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that a first render in a root holding no written file prints the
    path of the identity it wrote, and a second render prints nothing.
    """
    monkeypatch.chdir(tmp_path)
    render = load_task(".mise/tasks/docs/render.py")

    render.main()
    assert capsys.readouterr().out == "docs/.vitepress/lib/rendered/identity.json\n"

    render.main()
    assert capsys.readouterr().out == ""
