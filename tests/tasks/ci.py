"""
Pins the commands `repo:ci` runs, meaning `mise doctor project` first
and then one `mise run --continue-on-error --force` of every task the
pull-request rows run, and the status it exits with when the doctor or a
row fails.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pytest           import MonkeyPatch, mark, param, raises
from types            import ModuleType

from thermur.repo.checkout import Checkout


@mark.parametrize(
    ("doctor", "rows", "status", "ran"),
    [
        param(0, 0, 0, True, id="passing"),
        param(0, 3, 3, True, id="failing-row"),
        param(4, 0, 4, False, id="failing-doctor")
    ]
)
def test_the_rows_run_once_the_doctor_passes(
    checkout    : Checkout,
    doctor      : int,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch,
    ran         : bool,
    rows        : int,
    stand_in    : Callable[[str, str], StandIn],
    status      : int
):
    """
    Asserts that the task runs `mise doctor project`, stops with its status
    where it fails, and otherwise runs every row's task in one `mise run`
    under `--continue-on-error` and `--force`, exiting with that run's
    status.
    """
    monkeypatch.chdir(checkout.root)
    mise = stand_in("mise", f'[ "$1" = doctor ] && exit {doctor}\nexit {rows}')

    with raises(SystemExit) as exited:
        load_task(".mise/tasks/repo/ci.py").main()

    assert exited.value.code == status
    assert mise.commands == [
        "mise doctor project",
        *[f"mise run --continue-on-error --force {' ::: '.join(checkout.rows)}"] * ran
    ]
