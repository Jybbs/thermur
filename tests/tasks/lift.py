"""
Pins what `lock:lift` decides from the flags mise hands it through its usage
spec, over a clone of the checkout whose one lag lifts the `uv` mise pins:

- The table it prints, and the status `2` it exits with where a hold names
  no row, before it tries or writes anything
- The trial it runs under `--try`, failing with the major lift whose hold
  lets the rest pass, and the write it runs under `--write`, only once the
  trial passes where both are given
"""

from collections.abc  import Callable
from common.lags      import BREAKING, uv_lag
from common.stand_ins import StandIn
from contextlib       import nullcontext
from functools        import partial
from pytest           import CaptureFixture, MonkeyPatch, fixture, mark, param, raises
from types            import ModuleType

from thermur.repo.checkout import Checkout
from thermur.repo.lift     import Lift
from thermur.repo.schemas  import Config


@fixture
def lifted(
    clone       : Checkout,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
) -> Callable[..., None]:
    """
    Enters the clone, so a case runs the task over it.

    Returns:
        A runner taking each flag as mise hands it, by the variable naming
        it, which loads the task, holds every lift it builds to the one lag
        lifting the clone's `uv` pin, and runs its `main`.
    """
    monkeypatch.chdir(clone.root)

    def run(**flags: str):
        """
        Sets each variable `flags` names, then runs the task's `main`,
        loaded here so it prints through the streams the case captures.
        """
        for name, value in flags.items():
            monkeypatch.setenv(name, value)

        task = load_task(".mise/tasks/lock/lift.py")
        monkeypatch.setattr(task, "Lift", partial(Lift, lags=[uv_lag(clone)]))
        task.main()

    return run


@mark.parametrize(
    ("flags", "mise", "exited", "tried", "written"),
    [
        param({}, "exit 0", None, False, False, id="report"),
        param({"usage_write": "true"}, "exit 0", None, False, True, id="written"),
        param(
            {"usage_try": "true", "usage_write": "true"},
            "exit 0",
            None,
            True,
            True,
            id = "tried-and-written"
        ),
        param(
            {"usage_try": "true", "usage_write": "true"},
            BREAKING,
            "The lift fails its trial, and holding back `uv` lets the rest pass it",
            True,
            False,
            id = "isolated"
        ),
        param(
            {"usage_try": "true"},
            '[ "$*" = "run ci" ] && exit 1\nexit 0',
            "The lift fails its trial, and holding back no single major lift lets "
            "the rest pass it",
            True,
            False,
            id = "failing"
        )
    ],
    indirect = ["mise"]
)
def test_the_task_tries_and_writes_the_lift_as_its_flags_ask(
    capsys  : CaptureFixture[str],
    clone   : Checkout,
    exited  : str | None,
    flags   : dict[str, str],
    lifted  : Callable[..., None],
    mise    : StandIn,
    tried   : bool,
    written : bool
):
    """
    Asserts that the task prints the table, runs `mise ci` in a scratch
    clone only under `--try`, writes the lift into the checkout only under
    `--write` and, where both are given, only once the trial passes, and
    otherwise exits naming the major lift whose hold lets the rest pass, or
    naming none.
    """
    with raises(SystemExit) if exited else nullcontext() as stopped:
        lifted(**flags)

    assert capsys.readouterr().out.startswith("| **Pin** |")
    assert (stopped.value.code if exited else None) == exited
    assert ("mise run ci" in mise.commands) is tried
    assert (Config.read(clone.root).uv == uv_lag(clone).newest) is written


def test_a_hold_naming_no_row_stops_the_task_before_anything_runs(
    capsys : CaptureFixture[str],
    lifted : Callable[..., None],
    mise   : StandIn
):
    """
    Asserts that holds naming no row of the table stop the task with status
    `2` once the table prints, each named in order, before any trial or
    write runs.
    """
    with raises(SystemExit) as stopped:
        lifted(usage_hold="zeta uv alpha", usage_try="true", usage_write="true")

    assert (stopped.value.code, capsys.readouterr().err, mise.commands) == (
        2, "No row of the table names `alpha` and `zeta`\n",
        []
    )
