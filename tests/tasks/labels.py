"""
Pins what `repo:labels` sends GitHub and prints, meaning the command that
writes each label the registry declares, the read of every live label, the
live labels it lists for a rename or a deletion by hand, and the prompt that
stops it from writing anything unanswered.
"""

from collections.abc  import Callable
from pathlib          import Path
from pytest           import CaptureFixture, MonkeyPatch, TempPathFactory, fixture, mark, param, raises
from shlex            import join, quote
from subprocess       import CalledProcessError, DEVNULL, run
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

from thermur.repo.checkout import Checkout


def logged(log: Path) -> list[str]:
    """
    Reads each command the stand-in `gh` received out of `log`, quoted as a
    shell reads it.
    """
    return [
        join(["gh", *record.split("\0")[:-1]])
        for record in log.read_text(encoding="utf-8").splitlines()
    ]


@fixture
def log(
    stand_in         : Callable[[str, str], None],
    tmp_path_factory : TempPathFactory
) -> Path:
    """
    Puts a stand-in `gh` on the path that writes each command it receives
    to a log outside the copy of the checkout, one command a line with each
    argument closed by a NUL. It answers `api` with the live labels a `live`
    file beside the log names and exits with the status a `status` file
    there holds on every other command, `0` where none is written.

    Returns:
        The log's path.
    """
    folder = tmp_path_factory.mktemp("gh")
    stand_in(
        "gh",
        f"cd {quote(str(folder))}\n"
        "printf '%s\\0' \"$@\" >> log\n"
        "printf '\\n' >> log\n"
        '[ "$1" = api ] && exec cat live\n'
        "exit $(cat status 2>/dev/null || echo 0)"
    )
    return folder / "log"


@fixture
def synced(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    load_task   : Callable[[str], ModuleType],
    log         : Path,
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
        (log.parent / "live").write_text("".join(f"{name}\n" for name in live))
        load_task(".mise/tasks/repo/labels.py").main()
        return logged(log), capsys.readouterr().out.splitlines()

    return sync


def test_a_failed_write_stops_the_task_before_any_other(
    checkout : Checkout,
    log      : Path,
    synced   : Callable[[list[str]], tuple[list[str], list[str]]]
):
    """
    Asserts that a `label create` that `gh` fails stops the task there,
    sending neither the next label nor the read of the live labels.
    """
    (log.parent / "status").write_text("1\n")

    with raises(CalledProcessError):
        synced([])

    assert logged(log) == [join(checkout.labels.rows[0].command)]


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
    log      : Path
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
    assert not log.exists()
