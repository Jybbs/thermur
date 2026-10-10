"""
Pins the arguments each `py` task hands the formatter or pytest, meaning the
folders the formatter reads, the reports coverage writes, every argument a
reader adds, and the annotations `py:check` prints under GitHub Actions.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pytest           import Config, MonkeyPatch, mark, param
from subprocess       import run

FOLDERS = [".mise/tasks", "src", "tests"]


@mark.parametrize(
    ("task", "actions", "passed"),
    [
        param(
            "check",
            "true",
            ["check", "--output-format", "github", "--marker", *FOLDERS],
            id = "check-actions"
        ),
        param(
            "check",
            "",
            ["check", "--output-format", "text", "--marker", *FOLDERS],
            id = "check-empty"
        ),
        param(
            "check",
            None,
            ["check", "--output-format", "text", "--marker", *FOLDERS],
            id = "check-unset"
        ),
        param("format", None, ["format", "--marker", *FOLDERS], id="format"),
        param("test", None, ["--marker"], id="test"),
        param(
            "coverage",
            None,
            ["--cov", "--cov-report", "html", "--cov-report", "term", "--marker"],
            id = "coverage"
        )
    ]
)
def test_each_task_hands_its_program_every_argument(
    actions      : str | None,
    monkeypatch  : MonkeyPatch,
    passed       : list[str],
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn],
    task         : str
):
    """
    Asserts that each `py` task runs its program on the arguments it
    fixes followed by every argument a reader adds, the formatter over
    `.mise/tasks`, `src`, and `tests` and coverage writing its HTML and
    terminal reports, with `py:check` passing `--output-format github` where
    `GITHUB_ACTIONS` is set and `--output-format text` where it is empty
    or unset.
    """
    root = pytestconfig.rootpath
    for program in ("prose", "pytest"):
        stand_in(program, 'printf "%s\\n" "$@"')

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if actions is not None:
        monkeypatch.setenv("GITHUB_ACTIONS", actions)

    printed = run(
        [root / ".mise/tasks/py" / task, "--marker"],
        capture_output = True,
        check          = True,
        cwd            = root,
        text           = True
    ).stdout.splitlines()

    assert printed == passed
