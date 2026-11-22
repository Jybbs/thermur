"""
Pins what `docs:measure` writes and prints, meaning the trials and the sweep
of the run its argument names, each written where its page's charts read it
and only where the run saved it, failing on a run that saved neither.
"""

from collections.abc import Callable
from contextlib      import nullcontext
from json            import loads
from pathlib         import Path
from pytest          import CaptureFixture, MonkeyPatch, mark, param, raises
from types           import ModuleType

from thermur.docs import measured


@mark.parametrize(
    "tables",
    [
        param(("coupling_trial.nc",), id="trials"),
        param(("sweep.nc",), id="sweep"),
        param(("coupling_trial.nc", "sweep.nc"), id="both"),
        param((), id="neither")
    ]
)
def test_the_task_writes_what_the_run_named_saved_and_prints_where(
    capsys      : CaptureFixture[str],
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch,
    tables      : tuple[str, ...],
    tmp_path    : Path
):
    """
    Asserts that the task writes what `measured` and `swept` read from the
    run its first argument names into each chart file, creating its folder,
    for each table the run saved, and prints each file's path, exiting
    naming the run and writing nothing where it saved neither.
    """
    run = tmp_path / "runs" / "one"
    run.mkdir(parents=True)
    for table in tables:
        (run / table).touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["measure", "runs/one"])
    monkeypatch.setattr(measured, "measured", lambda run: {"trials": str(run)})
    monkeypatch.setattr(measured, "swept", lambda run: {"sweep": str(run)})

    with nullcontext() if tables else raises(
        SystemExit,
        match = "runs/one holds neither"
    ):
        load_task(".mise/tasks/docs/measure.py").main()

    written = {
        path: read
        for path, read, table in (
            (measured.MEASURED, {"trials": "runs/one"}, "coupling_trial.nc"),
            (measured.STILL_AIR, {"sweep": "runs/one"}, "sweep.nc")
        )
        if table in tables
    }
    assert capsys.readouterr().out == "".join(f"{path}\n" for path in written)
    assert {path: loads(path.read_text()) for path in written} == written
    assert all(
        path.exists() == (path in written)
        for path in (measured.MEASURED, measured.STILL_AIR)
    )
