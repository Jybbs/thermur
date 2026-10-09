"""
Pins the arguments and the network mode `repo:vet` hands zizmor, online with
annotations under GitHub Actions and offline in plain text elsewhere.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pytest           import Config, MonkeyPatch, mark, param
from subprocess       import run


@mark.parametrize(
    ("actions", "offline", "output"),
    [
        param("true", "false", "github", id="actions"),
        param("", "true", "plain", id="empty"),
        param(None, "true", "plain", id="unset")
    ]
)
def test_zizmor_audits_online_under_github_actions_alone(
    actions      : str | None,
    monkeypatch  : MonkeyPatch,
    offline      : str,
    output       : str,
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn]
):
    """
    Asserts that the task runs zizmor over the checkout with its cache under
    `.cache/zizmor`, online and printing annotations where `GITHUB_ACTIONS`
    is set, and offline in plain text where it is empty or unset.
    """
    stand_in("zizmor", 'echo "$ZIZMOR_OFFLINE"\nprintf "%s\\n" "$@"')
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if actions is not None:
        monkeypatch.setenv("GITHUB_ACTIONS", actions)

    assert run(
        [pytestconfig.rootpath / ".mise/tasks/repo/vet"],
        capture_output = True,
        check          = True,
        cwd            = pytestconfig.rootpath,
        text           = True
    ).stdout.splitlines() == [
        offline, "--cache-dir", ".cache/zizmor", "--format", output, "."
    ]
