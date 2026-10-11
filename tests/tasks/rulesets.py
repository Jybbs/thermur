"""
Pins what `repo:rulesets` sends GitHub and prints, meaning the read of every
live ruleset, the call updating or creating each ruleset, the call sending
each setting with the body it carries, the live rulesets it lists for a
deletion by hand, the validation that stops it before any call, and the
prompt that stops it from sending anything unanswered.
"""

from collections.abc  import Callable
from common.edits     import edit
from common.stand_ins import StandIn
from json             import dumps
from pathlib          import Path
from pytest           import CaptureFixture, MonkeyPatch, fixture, mark, param, raises
from subprocess       import CalledProcessError, DEVNULL, run
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

from thermur.repo.checkout import Checkout

LIVE = {"main branch protection": 24390266, "release tag protection": 24390267}


@fixture
def bodies(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    """
    Names the file the stand-in `gh` appends each body it reads to, one
    a line.
    """
    path = tmp_path / "bodies"
    monkeypatch.setenv("BODIES", str(path))
    return path


@fixture
def gh(bodies: Path, stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `gh` on the path that answers the `--jq` read of the
    live rulesets with the lines `LIVE` holds and exits with the status
    `READ_STATUS` names, and for any other call appends the body it reads
    from its standard input to the file `bodies` names and exits with the
    status `GH_STATUS` names, each `0` where none is set.
    """
    return stand_in(
        "gh",
        '[ "$2" = --jq ] && printf "%s" "$LIVE" && exit "${READ_STATUS:-0}"\n'
        '[ "$2" = --input ] && { cat; echo; } >> "$BODIES"\n'
        'exit "${GH_STATUS:-0}"'
    )


@fixture
def applied(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    gh          : StandIn,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
) -> Callable[[dict[str, int]], list[str]]:
    """
    Enters the copy of the checkout, so a case runs the task over it.

    Returns:
        A runner taking the id of each live ruleset by its name, which the
        stand-in `gh` reports, and returning the lines the task printed.
    """
    monkeypatch.chdir(checkout.root)

    def apply(live: dict[str, int]) -> list[str]:
        """
        Runs the task's `main` with `gh` reporting the rulesets in `live`,
        one name and id a line, as its `--jq` filter prints them.
        """
        monkeypatch.setenv("LIVE", "\n".join(map(dumps, live.items())))
        load_task(".mise/tasks/repo/rulesets.py").main()
        return capsys.readouterr().out.splitlines()

    return apply


def test_a_failed_call_stops_the_task_before_any_other(
    applied     : Callable[[dict[str, int]], list[str]],
    gh          : StandIn,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a call `gh` fails stops the task there, sending neither the
    next ruleset nor any setting.
    """
    monkeypatch.setenv("GH_STATUS", "1")

    with raises(CalledProcessError):
        applied(LIVE)

    assert gh.commands[1:] == [
        (
            "gh api --input - --method PUT --silent "
            "'repos/{owner}/{repo}/rulesets/24390266'"
        )
    ]


@mark.parametrize(
    ("path", "old", "new", "error"),
    [
        param(
            ".github/settings.toml",
            "has_wiki",
            "has_wiky",
            "repository.has_wiky\n  Extra inputs are not permitted",
            id = "settings"
        ),
        param(
            ".github/settings.toml",
            '= "all"',
            '= "al"',
            (
                "actions.allowed_actions\n  "
                "Input should be 'all', 'local_only' or 'selected'"
            ),
            id = "allowed-actions"
        ),
        param(
            ".github/settings.toml",
            '"read"',
            '"reed"',
            "token.default_workflow_permissions\n  Input should be 'read' or 'write'",
            id = "workflow-permissions"
        ),
        param(
            ".github/settings.toml",
            '{ status = "enabled" }',
            '{ status = "enabeld" }',
            "secret_scanning.status\n  Input should be 'disabled' or 'enabled'",
            id = "secret-scanning"
        ),
        param(
            ".github/settings.toml",
            'secret_scanning_push_protection = { status = "enabled" }',
            'secret_scanning_push_protection = { status = "enabeld" }',
            (
                "secret_scanning_push_protection.status\n  "
                "Input should be 'disabled' or 'enabled'"
            ),
            id = "push-protection"
        ),
        param(
            ".github/settings.toml",
            '"PR_TITLE"',
            '"PR_TILTE"',
            (
                "repository.squash_merge_commit_title\n  "
                "Input should be 'COMMIT_OR_PR_TITLE' or 'PR_TITLE'"
            ),
            id = "squash-title"
        ),
        param(
            ".github/settings.toml",
            '"PR_BODY"',
            '"PR_BODDY"',
            (
                "repository.squash_merge_commit_message\n  "
                "Input should be 'BLANK', 'COMMIT_MESSAGES' or 'PR_BODY'"
            ),
            id = "squash-message"
        ),
        param(
            ".github/rulesets/tags.json",
            '"enforcement"',
            '"enforcment"',
            "enforcment\n  Extra inputs are not permitted",
            id = "ruleset"
        ),
        param(
            ".github/rulesets/tags.json",
            '"enforcement": "active"',
            '"enforcement": "actve"',
            "enforcement\n  Input should be 'active', 'disabled' or 'evaluate'",
            id = "enforcement"
        ),
        param(
            ".github/rulesets/tags.json",
            '"target": "tag"',
            '"target": "tags"',
            "target\n  Input should be 'branch', 'push' or 'tag'",
            id = "target"
        ),
        param(
            ".github/rulesets/tags.json",
            '{ "type": "deletion" }',
            '{ "type": "deletoin" }',
            "rules.1.type\n  Input should be 'branch_name_pattern'",
            id = "rule-type"
        ),
        param(
            ".github/rulesets/tags.json",
            '"bypass_actors": []',
            '"bypass_actors": [{ "actor_id": 1, "actor_type": "Teams" }]',
            (
                "bypass_actors.0.actor_type\n  Input should be 'DeployKey', "
                "'Integration', 'OrganizationAdmin', 'RepositoryRole', 'Team' or 'User'"
            ),
            id = "actor-type"
        ),
        param(
            ".github/rulesets/tags.json",
            '"bypass_actors": []',
            (
                '"bypass_actors": '
                '[{ "actor_id": 1, "actor_type": "Team", "bypass_mode": "alway" }]'
            ),
            (
                "bypass_actors.0.bypass_mode\n  "
                "Input should be 'always', 'exempt' or 'pull_request'"
            ),
            id = "bypass-mode"
        ),
        param(
            ".github/rulesets/tags.json",
            '"bypass_actors": []',
            (
                '"bypass_actors": '
                '[{ "actor_id": 1, "actor_type": "Team", "bypas_mode": "always" }]'
            ),
            "bypass_actors.0.bypas_mode\n  Extra inputs are not permitted",
            id = "bypass-actor"
        ),
        param(
            ".github/rulesets/tags.json",
            '"bypass_actors": []',
            '"bypass_actors": [{ "actor_id": "one", "actor_type": "Team" }]',
            "bypass_actors.0.actor_id\n  Input should be a valid integer",
            id = "actor-id"
        ),
        param(
            ".github/rulesets/main.json",
            '["squash"]',
            '["sqaush"]',
            (
                "rules.0.parameters.allowed_merge_methods.0\n  "
                "Input should be 'merge', 'rebase' or 'squash'"
            ),
            id = "merge-method"
        ),
        param(
            ".github/rulesets/main.json",
            '"parameters"',
            '"parameter"',
            "rules.0.parameter\n  Extra inputs are not permitted",
            id = "rule"
        ),
        param(
            ".github/rulesets/main.json",
            '"integration_id"',
            '"integration"',
            "required_status_checks.0.integration\n  Extra inputs are not permitted",
            id = "required-check"
        ),
        param(
            ".github/rulesets/tags.json",
            '"target": "tag"\n',
            '"target": "tag",\n',
            "Illegal trailing comma before end of object",
            id = "trailing-comma"
        ),
        param(
            ".github/rulesets/tags.json",
            '"target": "tag"\n',
            '"target": "tag",\n  "target": "branch"\n',
            "holds `target` more than once",
            id = "repeated-key"
        ),
        param(
            "pyproject.toml",
            "keywords        = [",
            "topics          = [",
            "project.keywords\n  Field required",
            id = "manifest"
        )
    ]
)
def test_a_file_failing_validation_fails_before_anything_reaches_github(
    applied  : Callable[[dict[str, int]], list[str]],
    checkout : Checkout,
    error    : str,
    gh       : StandIn,
    new      : str,
    old      : str,
    path     : str
):
    """
    Asserts that a misspelled settings key, a settings or ruleset value
    outside the closed set its field takes, a key the rulesets API does not
    take at any depth the ruleset validates, a ruleset file that is not JSON
    or that repeats a key, and a manifest leaving out `[project].keywords`
    each end the task on the validation error naming it, before the task
    reads or sends anything.
    """
    edit(checkout, path, new=new, old=old)

    with raises(SystemExit) as exited:
        applied(LIVE)

    assert error in str(exited.value.code)
    assert gh.commands == []


def test_a_failed_read_of_the_live_rulesets_stops_the_task(
    applied     : Callable[[dict[str, int]], list[str]],
    gh          : StandIn,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a read of the live rulesets that `gh` fails stops the
    task before any write, rather than reading as no live ruleset, which
    would send every ruleset through `POST` in place of the update its live
    ruleset takes.
    """
    monkeypatch.setenv("READ_STATUS", "1")

    with raises(CalledProcessError):
        applied(LIVE)

    assert gh.commands == [
        "gh api --jq '.[] | [.name, .id]' --paginate "
        "'repos/{owner}/{repo}/rulesets?includes_parents=false'"
    ]


def test_a_ruleset_no_live_ruleset_carries_is_created(
    applied : Callable[[dict[str, int]], list[str]],
    gh      : StandIn
):
    """
    Asserts that a ruleset whose name no live ruleset carries is created
    through `POST`, while one whose name a live ruleset carries is updated
    in place through `PUT` on that ruleset's id.
    """
    applied({"release tag protection": 24390267})

    assert gh.commands[1:3] == [
        "gh api --input - --method POST --silent 'repos/{owner}/{repo}/rulesets'",
        (
            "gh api --input - --method PUT --silent "
            "'repos/{owner}/{repo}/rulesets/24390267'"
        )
    ]


def test_each_ruleset_and_setting_is_sent_with_its_body(
    applied  : Callable[[dict[str, int]], list[str]],
    bodies   : Path,
    gh       : StandIn,
    snapshot : SnapshotAssertion
):
    """
    Asserts that the task reads the live rulesets through `--paginate`,
    updates each ruleset in place, and then sends the repository's settings,
    the Actions permissions, the token's defaults, each toggle, and last the
    manifest's description and its keywords as the topics, each call beside
    the body it sends.
    """
    applied(LIVE)

    sent = iter(bodies.read_text().splitlines())

    assert "\n".join(
        line
        for command in gh.commands
        for line in [command, *([next(sent)] if "--input -" in command else [])]
    ) == snapshot


@mark.parametrize(
    ("older", "printed"),
    [
        param(
            {"draft protection": 2, "old protection": 1},
            [
                "No file under `.github/rulesets/` names these live rulesets, left "
                "for a deletion by hand:",
                "draft protection",
                "old protection"
            ],
            id = "older-rulesets"
        ),
        param({}, [], id = "all-named")
    ]
)
def test_each_live_ruleset_no_file_names_is_listed(
    applied : Callable[[dict[str, int]], list[str]],
    older   : dict[str, int],
    printed : list[str]
):
    """
    Asserts that the task lists each live ruleset no file names, in name
    order beneath a line naming what they are, and prints nothing once a
    file names every live ruleset.
    """
    assert applied(LIVE | older) == printed


def test_nothing_is_sent_without_an_answer_to_the_prompt(
    checkout : Checkout,
    gh       : StandIn
):
    """
    Asserts that mise stops the task on its prompt where no terminal can
    answer it, so the stand-in `gh` receives no command at all.
    """
    prompted = run(
        ["mise", "run", "repo:rulesets"],
        capture_output = True,
        cwd            = checkout.root,
        stdin          = DEVNULL,
        text           = True
    )

    assert prompted.returncode == 1
    assert "requires confirmation" in prompted.stderr
    assert gh.commands == []


def test_two_rulesets_sharing_a_name_fail_before_anything_reaches_github(
    applied  : Callable[[dict[str, int]], list[str]],
    checkout : Checkout,
    gh       : StandIn
):
    """
    Asserts that two files under `.github/rulesets/` carrying one name end
    the task naming it, before the task reads or sends anything, since the
    task matches each file to a live ruleset by the name it carries.
    """
    rulesets = checkout.root / ".github/rulesets"
    (rulesets / "tags.json").copy(rulesets / "copy.json")

    with raises(SystemExit) as exited:
        applied(LIVE)

    assert exited.value.code == (
        "More than one file under `.github/rulesets/` names each of these "
        "rulesets, which the task matches to a live ruleset by name: release tag "
        "protection"
    )
    assert gh.commands == []
