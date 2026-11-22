"""
Pins the path `gha:cut` takes for each answer the repository's releases give
it, meaning the draft it creates, the draft whose notes it updates, and the
published release it fails on, the outputs and the exit status it leaves for
the gate, and the commands it sends GitHub on each path.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from json             import dumps
from pathlib          import Path
from pytest           import (
    CaptureFixture, Config, MonkeyPatch, fixture, mark, param, raises
)
from subprocess       import CalledProcessError, run
from syrupy.assertion import SnapshotAssertion
from sys              import executable
from types            import ModuleType

from thermur.repo.schemas import Workflow

DRAFTED = "https://github.com/Jybbs/thermur/releases/tag/untagged-1"
UPDATED = "https://github.com/Jybbs/thermur/releases/tag/untagged-2"


@fixture
def cut(
    gh          : StandIn,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
) -> ModuleType:
    """
    Enters a folder holding a `pyproject.toml` at `0.2.0`, behind the
    stand-in `gh`.

    Returns:
        The task script, imported as a module.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
    return load_task(".mise/tasks/gha/cut.py")


@fixture
def gh(stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `gh` on the path that prints generated notes for
    the generate-notes call, writes the body a `PATCH` receives on stdin
    to `patched.json` and prints the updated URL, prints the releases
    `RELEASES` holds for any other `api` call and a URL for a `release` one,
    and exits with the status `GH_STATUS` names.
    """
    return stand_in(
        "gh",
        'case "$*" in\n'
        '  *releases/generate-notes*) echo \'{"body":"notes"}\' ;;\n'
        f"  *PATCH*) cat > patched.json; echo {UPDATED} ;;\n"
        '  api*) printf "%s" "$RELEASES" ;;\n'
        f"  release*) echo {DRAFTED} ;;\n"
        "esac\n"
        'exit "${GH_STATUS:-0}"'
    )


def release(*, draft: bool, url: str) -> str:
    """
    Writes one release as the line the task's `--jq` filter prints for it.
    """
    return dumps({"draft": draft, "id": 7, "url": url})


@mark.parametrize(
    ("releases", "outputs", "status"),
    [
        param(
            "",
            ["state=created", f"url={DRAFTED}", "version=0.2.0"],
            0,
            id = "created"
        ),
        param(
            release(draft=True, url="https://u/draft"),
            ["state=updated", f"url={UPDATED}", "version=0.2.0"],
            0,
            id = "updated"
        ),
        param(
            release(draft=False, url="https://u/published"),
            ["state=published", "url=https://u/published", "version=0.2.0"],
            1,
            id = "published"
        ),
        param(
            f"{release(draft=True, url='https://u/draft')}\n"
            f"{release(draft=False, url='https://u/published')}",
            ["state=published", "url=https://u/published", "version=0.2.0"],
            1,
            id = "published-beside-draft"
        )
    ]
)
def test_the_cut_takes_the_path_the_releases_call_for(
    cut         : ModuleType,
    gh          : StandIn,
    monkeypatch : MonkeyPatch,
    outputs     : list[str],
    releases    : str,
    snapshot    : SnapshotAssertion,
    status      : int,
    tmp_path    : Path
):
    """
    Asserts that the cut creates the draft where no release carries the
    version, updates the notes of a draft carrying it, and fails where a
    published release carries it, beside a draft or alone, writing the path,
    the URL, and the version to the job's outputs on every path.
    """
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("RELEASES", releases)

    assert cut.Draft.read().emit() == status
    assert (tmp_path / "outputs").read_text().splitlines() == outputs
    assert "\n".join(gh.commands) == snapshot


def test_a_published_version_fails_naming_the_bump(
    capsys      : CaptureFixture[str],
    cut         : ModuleType,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the published path prints an error annotation naming the
    version and `mise bump` as the fix.
    """
    monkeypatch.setenv("RELEASES", release(draft=False, url="https://u/published"))

    cut.Draft(version="0.2.0").cut()

    assert capsys.readouterr().out.startswith(
        "::error::0.2.0 is already published, and `mise bump`"
    )


def test_a_release_listing_that_fails_fails_the_cut(
    cut         : ModuleType,
    gh          : StandIn,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a release listing failing for any reason fails the cut
    rather than reading as no release, which would create a second draft.
    """
    monkeypatch.setenv("GH_STATUS", "1")

    with raises(CalledProcessError):
        cut.Draft(version="0.2.0").cut()

    assert len(gh.commands) == 1


def test_an_updated_draft_takes_the_notes_github_generates_now(
    cut         : ModuleType,
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that the `PATCH` updating a draft sends only the body the
    generate-notes endpoint returned, leaving the draft's title as it was.
    """
    monkeypatch.setenv("RELEASES", release(draft=True, url="https://u/draft"))

    cut.Draft(version="0.2.0").cut()

    assert (tmp_path / "patched.json").read_text() == '{"body":"notes"}'


def test_the_draft_runs_on_every_push_to_main_and_on_dispatch(pytestconfig: Config):
    """
    Asserts that `🐦‍⬛ Draft` runs on `workflow_dispatch` and on every push
    to `main`, whatever files the push touches.
    """
    draft = Workflow.read(pytestconfig.rootpath, Path(".github/workflows/draft.yml"))

    assert (sorted(draft.events), draft.events["push"].paths) == (
        ["push", "workflow_dispatch"],
        ()
    )


def test_the_outputs_print_where_no_runner_collects_them(
    capsys : CaptureFixture[str],
    cut    : ModuleType
):
    """
    Asserts that the outputs print to stdout where `GITHUB_OUTPUT` is unset,
    as on a laptop.
    """
    cut.Draft.read().emit()

    assert capsys.readouterr().out.splitlines() == [
        "state=created",
        f"url={DRAFTED}",
        "version=0.2.0"
    ]


@mark.parametrize(
    ("releases", "outputs", "status"),
    [
        param(
            "",
            ["state=created", f"url={DRAFTED}", "version=0.2.0"],
            0,
            id = "created"
        ),
        param(
            release(draft=False, url="https://u/published"),
            ["state=published", "url=https://u/published", "version=0.2.0"],
            1,
            id = "published"
        )
    ]
)
def test_the_task_exits_with_the_status_its_path_calls_for(
    cut          : ModuleType,
    monkeypatch  : MonkeyPatch,
    outputs      : list[str],
    pytestconfig : Config,
    releases     : str,
    status       : int
):
    """
    Asserts that the task, run as its own process, prints the path, the URL,
    and the version and exits with status `0` on the created path and `1` on
    the published one, which fails the draft job and the gate after it.
    """
    monkeypatch.setenv("RELEASES", releases)

    task = run(
        [executable, pytestconfig.rootpath / ".mise/tasks/gha/cut.py"],
        capture_output = True,
        text           = True
    )

    assert (task.returncode, task.stdout.splitlines()[-3:]) == (status, outputs)
