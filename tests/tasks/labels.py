"""
Pins what `repo:labels` sends GitHub and prints, meaning the command that
writes each label the registry declares, the read of every live label, the
live labels it lists for a rename or a deletion by hand, and the prompt that
stops it from writing anything unanswered.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pytest           import CaptureFixture, MonkeyPatch, fixture, mark, param, raises
from shlex            import join
from subprocess       import CalledProcessError, DEVNULL, run
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

from thermur.repo.checkout import Checkout


@fixture
def gh(stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `gh` on the path that answers `api` with the live labels
    `LIVE` holds and exits with the status `GH_STATUS` names on every other
    command, `0` where none is set.
    """
    return stand_in(
        "gh",
        '[ "$1" = api ] && printf "%s" "$LIVE" && exit 0\n'
        'exit "${GH_STATUS:-0}"'
    )


@fixture
def synced(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    gh          : StandIn,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
) -> Callable[[list[str]], tuple[list[str], list[str]]]:
    """
    Enters the copy of the checkout, so a case runs the task over it.

    Returns:
        A runner taking the live labels the stand-in `gh` reports and
        returning the commands it received and the lines the task printed.
    """
    monkeypatch.chdir(checkout.root)

    def sync(live: list[str]) -> tuple[list[str], list[str]]:
        """
        Runs the task's `main` with `gh` answering `api` with `live`, one
        name a line.
        """
        monkeypatch.setenv("LIVE", "\n".join(live))
        load_task(".mise/tasks/repo/labels.py").main()
        return gh.commands, capsys.readouterr().out.splitlines()

    return sync


def test_a_failed_write_stops_the_task_before_any_other(
    checkout    : Checkout,
    gh          : StandIn,
    monkeypatch : MonkeyPatch,
    synced      : Callable[[list[str]], tuple[list[str], list[str]]]
):
    """
    Asserts that a `label create` that `gh` fails stops the task there,
    sending neither the next label nor the read of the live labels.
    """
    monkeypatch.setenv("GH_STATUS", "1")

    with raises(CalledProcessError):
        synced([])

    assert gh.commands == [join(checkout.labels.rows[0].command)]


def test_each_label_is_written_then_every_live_label_read(
    checkout : Checkout,
    snapshot : SnapshotAssertion,
    synced   : Callable[[list[str]], tuple[list[str], list[str]]]
):
    """
    Asserts that the task writes every label the registry declares through
    `gh label create --force`, in the registry's order, then reads every
    live label across every page.
    """
    assert "\n".join(synced(checkout.labels.names)[0]) == snapshot


@mark.parametrize(
    ("older", "printed"),
    [
        param(
            ["✨feature", "🐞bug"],
            [
                "`.github/labels.toml` omits these live labels, left for a rename or "
                "deletion by hand:",
                "✨feature",
                "🐞bug"
            ],
            id = "older-labels"
        ),
        param([], [], id="all-declared")
    ]
)
def test_each_live_label_the_registry_omits_is_listed(
    checkout : Checkout,
    older    : list[str],
    printed  : list[str],
    synced   : Callable[[list[str]], tuple[list[str], list[str]]]
):
    """
    Asserts that the task lists each live label the registry omits beneath a
    line naming what they are, and prints nothing once the registry declares
    every live label.
    """
    assert synced([*checkout.labels.names, *older])[1] == printed


def test_nothing_is_written_without_an_answer_to_the_prompt(
    checkout : Checkout,
    gh       : StandIn
):
    """
    Asserts that mise stops the task on its prompt where no terminal can
    answer it, so the stand-in `gh` receives no command at all.
    """
    prompted = run(
        ["mise", "run", "repo:labels"],
        capture_output = True,
        cwd            = checkout.root,
        stdin          = DEVNULL,
        text           = True
    )

    assert prompted.returncode == 1
    assert "requires confirmation" in prompted.stderr
    assert gh.commands == []
