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
from pytest           import CaptureFixture, MonkeyPatch, fixture, mark, param, raises
from subprocess       import CalledProcessError
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

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
        "::error::0.2.0 is already published, so `mise bump`"
    )


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
        cut.Draft(version="0.2.0").cut()


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
