"""
Pins the path `gha:cut` takes for each answer the repository's releases give
it, meaning the cut a push leaving the version where it stood skips, the
draft it creates, keeps, or leaves to a published release, the outputs it
writes for the gate, and the commands it sends GitHub on each path.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from json             import dumps
from pathlib          import Path
from pytest           import CaptureFixture, MonkeyPatch, fixture, mark, param, raises
from subprocess       import CalledProcessError
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

DRAFTED = "https://github.com/Jybbs/thermur/releases/tag/untagged-1"


@fixture
def cut(
    gh          : StandIn,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch,
    stand_in    : Callable[[str, str], StandIn],
    tmp_path    : Path
) -> ModuleType:
    """
    Enters a folder holding a `pyproject.toml` at `0.2.0`, behind a stand-in
    `git` printing a manifest at the version `PREVIOUS` names for `HEAD~1`
    and the stand-in `gh`.

    Returns:
        The task script, imported as a module.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
    stand_in("git", 'printf \'[project]\\nversion = "%s"\\n\' "$PREVIOUS"')
    return load_task(".mise/tasks/gha/cut.py")


@fixture
def gh(stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `gh` on the path that prints the releases `RELEASES`
    holds for an `api` call and a URL for a `release` one, and exits with
    the status `GH_STATUS` names.
    """
    return stand_in(
        "gh",
        '[ "$1" = api ] && printf "%s" "$RELEASES"\n'
        f'[ "$1" = release ] && echo {DRAFTED}\n'
        'exit "${GH_STATUS:-0}"'
    )


def release(*, draft: bool, url: str) -> str:
    """
    Writes one release as the line the task's `--jq` filter prints for it.
    """
    return dumps({"draft": draft, "url": url})


@mark.parametrize(
    ("event", "previous", "releases", "outputs"),
    [
        param(
            "push",
            "0.2.0",
            "",
            ["state=unchanged", "url=", "version=0.2.0"],
            id = "unchanged"
        ),
        param(
            "push",
            "0.1.0",
            "",
            [
                "state=created",
                f"url={DRAFTED}",
                "version=0.2.0"
            ],
            id = "created"
        ),
        param(
            "workflow_dispatch",
            "0.2.0",
            "",
            [
                "state=created",
                f"url={DRAFTED}",
                "version=0.2.0"
            ],
            id = "dispatched"
        ),
        param(
            "push",
            "0.1.0",
            release(draft=True, url="https://u/draft"),
            ["state=kept", "url=https://u/draft", "version=0.2.0"],
            id = "kept"
        ),
        param(
            "push",
            "0.1.0",
            f"{release(draft=True, url='https://u/draft')}\n"
            f"{release(draft=False, url='https://u/published')}",
            ["state=skipped", "url=https://u/published", "version=0.2.0"],
            id = "skipped"
        )
    ]
)
def test_the_cut_takes_the_path_the_releases_call_for(
    cut         : ModuleType,
    event       : str,
    gh          : StandIn,
    monkeypatch : MonkeyPatch,
    outputs     : list[str],
    previous    : str,
    releases    : str,
    snapshot    : SnapshotAssertion,
    tmp_path    : Path
):
    """
    Asserts that a push leaving the version where it stood cuts nothing and
    asks GitHub nothing, that a moved version or a `workflow_dispatch` run
    creates the draft where no release carries the version, keeps a draft
    carrying it, and skips where a published release carries it, writing the
    path, the URL, and the version to the job's outputs.
    """
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("PREVIOUS", previous)
    monkeypatch.setenv("RELEASES", releases)

    cut.Draft.read().emit()

    assert (tmp_path / "outputs").read_text().splitlines() == outputs
    assert "\n".join(gh.commands) == snapshot


def test_a_published_version_warns_rather_than_failing(
    capsys      : CaptureFixture[str],
    cut         : ModuleType,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the skip prints a warning annotation naming the version, so
    the workflow run passes while its summary flags the skip.
    """
    monkeypatch.setenv("RELEASES", release(draft=False, url="https://u/published"))

    assert cut.Draft(previous=None, version="0.2.0").cut() == (
        "skipped", "https://u/published"
    )
    assert capsys.readouterr().out.startswith("::warning::0.2.0 is already published")


def test_a_release_listing_that_fails_fails_the_cut(
    cut         : ModuleType,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a release listing failing for any reason fails the cut
    rather than reading as no release, which would create a second draft.
    """
    monkeypatch.setenv("GH_STATUS", "1")

    with raises(CalledProcessError):
        cut.Draft(previous=None, version="0.2.0").cut()


def test_the_outputs_print_where_no_runner_collects_them(
    capsys      : CaptureFixture[str],
    cut         : ModuleType,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the outputs print to stdout where `GITHUB_OUTPUT` is unset,
    as on a laptop.
    """
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("PREVIOUS", "0.2.0")

    cut.Draft.read().emit()

    assert capsys.readouterr().out.splitlines() == [
        "state=unchanged", "url=", "version=0.2.0"
    ]
