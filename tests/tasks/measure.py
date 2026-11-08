"""
Pins what `docs:measure` writes and prints, meaning the trials of the run
its argument names, written where the rules page's charts read them.
"""

from collections.abc import Callable
from json            import loads
from pathlib         import Path
from pytest          import CaptureFixture, MonkeyPatch
from types           import ModuleType

from thermur.docs import measured


def test_the_task_writes_the_run_named_and_prints_where(
    capsys      : CaptureFixture[str],
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that the task writes what `measured` reads from the run its
    first argument names into the charts' file, creating its folder, and
    prints the file's path.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["measure", "runs/one"])
    monkeypatch.setattr(measured, "measured", lambda run: {"run": str(run)})

    load_task(".mise/tasks/docs/measure.py").main()

    assert capsys.readouterr().out == f"{measured.MEASURED}\n"
    assert loads(measured.MEASURED.read_text()) == {"run": "runs/one"}
