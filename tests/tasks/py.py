"""
Pins the output format `py:check` passes the formatter, the annotations
GitHub Actions reads under a workflow run and plain text everywhere else.
"""

from collections.abc import Callable
from pytest          import Config, MonkeyPatch, mark, param
from subprocess      import run


@mark.parametrize(
    ("actions", "output"),
    [
        param("true", "github", id="actions"),
        param("", "text", id="empty"),
        param(None, "text", id="unset")
    ]
)
def test_the_check_prints_annotations_under_github_actions(
    actions      : str | None,
    monkeypatch  : MonkeyPatch,
    output       : str,
    pytestconfig : Config,
    stand_in     : Callable[[str, str], None]
):
    """
    Asserts that `py:check` passes the formatter `--output-format github`
    where `GITHUB_ACTIONS` is set and `--output-format text` where it is
    empty or unset, ahead of the folders it checks.
    """
    root = pytestconfig.rootpath
    stand_in("prose", 'printf "%s\\n" "$@"')
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if actions is not None:
        monkeypatch.setenv("GITHUB_ACTIONS", actions)

    passed = run(
        [root / ".mise/tasks/py/check"],
        capture_output = True,
        check          = True,
        cwd            = root,
        text           = True
    ).stdout.splitlines()

    assert passed == ["check", "--output-format", output, ".mise/tasks", "src", "tests"]
