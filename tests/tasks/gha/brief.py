"""
Pins what `gha:brief` writes and the verdict it exits with, meaning the
summary each workflow's template renders, the variables it reads the commit
and the branch from, where it writes the summary, and the status it returns
for the results of the jobs its gate waited on.
"""

from collections.abc  import Callable
from dataclasses      import replace
from json             import dumps
from pathlib          import Path
from pytest           import CaptureFixture, Config, MonkeyPatch, fixture, mark, param
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType

from thermur.repo.schemas import Workflow

SHA = "0123456789abcdef0123456789abcdef01234567"


@fixture
def readout(
    load_task    : Callable[[str], ModuleType],
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config
) -> Callable[..., object]:
    """
    Enters the worktree root, whose `.github/scripts/summaries/` the task
    reads, and sets the variables a runner sets for a workflow run, on
    a pull request's merge ref for a workflow a pull request runs and on
    `main` for any other.

    Returns:
        A builder reading the gate of a run of the workflow file named, the
        `needs` context given, and the branch where one is given, as the
        task reads it.
    """
    monkeypatch.chdir(pytestconfig.rootpath)
    for name, value in {
        "GITHUB_REPOSITORY" : "Jybbs/thermur",
        "GITHUB_SERVER_URL" : "https://github.com",
        "GITHUB_SHA"        : SHA
    }.items():
        monkeypatch.setenv(name, value)

    module = load_task(".mise/tasks/gha/brief.py")

    def build(file: str, needs: dict, branch: str | None = None) -> object:
        """
        Reads the gate of a run of `.github/workflows/<file>` waiting on
        the jobs `needs` describes, under the name the workflow declares, on
        `branch` where one is given.
        """
        path     = Path(".github/workflows", file)
        workflow = Workflow.read(pytestconfig.rootpath, path)
        ref      = (
            f"heads/{branch}" if branch
            else "pull/55/merge" if workflow.on_pull_request
            else "heads/main"
        )
        monkeypatch.setenv("GITHUB_REF_NAME", ref.partition("/")[2])
        monkeypatch.setenv("GITHUB_WORKFLOW", workflow.name)
        monkeypatch.setenv("GITHUB_WORKFLOW_REF", f"Jybbs/thermur/{path}@refs/{ref}")
        monkeypatch.setenv("NEEDS", dumps(needs))
        return module.Readout.running()

    return build


def job(result: str = "success", **outputs: str) -> dict:
    """
    Describes one job the gate waited on as the `needs` context holds it.
    """
    return {"outputs": outputs, "result": result}


@mark.parametrize(
    ("file", "needs", "branch"),
    [
        param(
            "ci.yml",
            {"check": job(coverage="https://example.invalid/1")},
            None,
            id = "ci"
        ),
        param("ci.yml", {"check": job("failure")}, None, id="ci-failed"),
        param(
            "draft.yml",
            {"cut": job(state="created", url="https://u/d", version="0.2.0")},
            None,
            id = "draft-created"
        ),
        param(
            "draft.yml",
            {"cut": job(state="kept", url="https://u/d", version="0.2.0")},
            None,
            id = "draft-kept"
        ),
        param(
            "draft.yml",
            {"cut": job(state="skipped", url="https://u/r", version="0.1.0")},
            None,
            id = "draft-skipped"
        ),
        param(
            "draft.yml",
            {"cut": job(state="unchanged", url="", version="0.1.0")},
            None,
            id = "draft-unchanged"
        ),
        param("draft.yml", {"cut": job("failure")}, None, id="draft-failed"),
        param(
            "pages.yml",
            {"deploy": job(url="https://0a1b2c3d.thermur.pages.dev"), "press": job()},
            None,
            id = "pages"
        ),
        param(
            "pages.yml",
            {"deploy": job("skipped"), "press": job()},
            "55/pages-preview",
            id = "pages-elsewhere"
        ),
        param(
            "pages.yml",
            {"deploy": job("failure"), "press": job()},
            None,
            id = "pages-failed"
        ),
        param(
            "pages.yml",
            {"deploy": job("skipped"), "press": job("failure")},
            None,
            id = "pages-unpressed"
        ),
        param(
            "site.yml",
            {"check": job(**{"site-coverage": "https://example.invalid/2"})},
            None,
            id = "site"
        ),
        param(
            "site.yml",
            {"check": job("failure", **{"site-coverage": ""})},
            None,
            id = "site-failed"
        ),
        param("warm.yml", {"kit": job(), "prune": job()}, None, id="warm"),
        param(
            "warm.yml",
            {"kit": job(), "prune": job("failure")},
            None,
            id = "warm-failed"
        )
    ]
)
def test_each_workflow_renders_its_summary(
    branch   : str | None,
    file     : str,
    needs    : dict,
    readout  : Callable[..., object],
    snapshot : SnapshotAssertion
):
    """
    Asserts that a workflow file with a template of its own renders that
    template, naming the coverage report a coverage row attached and none
    where the row left its output empty, the path the draft cut took, or the
    address the deploy published, and that any other renders `base.md.j2`.
    """
    assert readout(file, needs, branch).summary == snapshot


def test_the_gate_read_from_the_runner_hashes_alike_across_reads(
    readout: Callable[[str, dict], object]
):
    """
    Asserts that the gate the task reads hashes, every object of its `needs`
    context read as a `frozendict`, each job's outputs included, and that
    two reads of one context hash alike.
    """
    needs = {"check": job(coverage="https://github.com/Jybbs/thermur/actions")}

    assert hash(readout("ci.yml", needs)) == hash(readout("ci.yml", needs))


@mark.parametrize(
    ("file", "needs"),
    [
        param("ci.yml", {"check": job()}, id="ci"),
        param("draft.yml", {"cut": job("failure")}, id="draft"),
        param("pages.yml", {"deploy": job("skipped"), "press": job()}, id="pages"),
        param("site.yml", {"check": job()}, id="site"),
        param("warm.yml", {"kit": job()}, id="warm")
    ]
)
def test_each_workflow_s_name_opens_on_the_glyph_its_heading_opens_on(
    file    : str,
    needs   : dict,
    readout : Callable[[str, dict], object]
):
    """
    Asserts that each workflow's name opens on the glyph the heading of its
    summary opens on, so a workflow renamed onto another mark fails rather
    than rendering a heading its name no longer matches.
    """
    gate = readout(file, needs)

    assert gate.summary.partition("\n")[0] == (
        f"## {gate.workflow.replace(' ', ' Thermur ', 1)}"
    )


@mark.parametrize(
    ("results", "status"),
    [
        param(["success", "success"], 0, id="passed"),
        param(["success", "failure"], 1, id="failed"),
        param(["success", "cancelled"], 1, id="cancelled"),
        param(["success", "skipped"], 0, id="skipped")
    ]
)
def test_the_verdict_passes_only_where_every_job_succeeded_or_was_skipped(
    readout : Callable[[str, dict], object],
    results : list[str],
    status  : int
):
    """
    Asserts that the task exits `0` only where every job the gate waited on
    succeeded or was skipped, as the deploy job is on a run outside `main`,
    and `1` where any failed or was cancelled.
    """
    needs = {f"job{index}": job(result) for index, result in enumerate(results)}

    assert readout("warm.yml", needs).write() == status


def test_the_heading_keeps_every_word_after_the_glyph(
    readout: Callable[[str, dict], object]
):
    """
    Asserts that the summary's heading names the workflow by every word its
    name carries after its glyph, so a name of two words keeps both.
    """
    gate = replace(readout("warm.yml", {"kit": job()}), workflow="🐦‍⬛ Pages Deploy")

    assert gate.summary.partition("\n")[0] == "## 🐦‍⬛ Thermur Pages Deploy"


def test_the_summary_appends_to_the_runner_s_summary_file(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch,
    readout     : Callable[[str, dict], object],
    tmp_path    : Path
):
    """
    Asserts that the summary is appended to the file `GITHUB_STEP_SUMMARY`
    names, after whatever it already holds, and printed nowhere.
    """
    page = tmp_path / "summary.md"
    page.write_text("earlier\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(page))
    gate = readout("warm.yml", {"kit": job()})

    gate.write()

    assert page.read_text() == f"earlier\n{gate.summary}\n"
    assert capsys.readouterr().out == ""


@mark.parametrize(
    ("file", "variables", "commit", "ref"),
    [
        param("warm.yml", {}, SHA, "main", id="runner"),
        param(
            "ci.yml",
            {"COMMIT": "fedcba9876543210", "GITHUB_HEAD_REF": "55/ci-brief-gate"},
            "fedcba9876543210",
            "55/ci-brief-gate",
            id = "pull-request"
        ),
        param(
            "warm.yml",
            {"COMMIT": "", "GITHUB_HEAD_REF": ""},
            SHA,
            "main",
            id = "empty"
        )
    ]
)
def test_the_commit_and_the_branch_prefer_the_pull_request_s_head(
    commit      : str,
    file        : str,
    monkeypatch : MonkeyPatch,
    readout     : Callable[[str, dict], object],
    ref         : str,
    variables   : dict[str, str]
):
    """
    Asserts that the gate names the commit `COMMIT` sets and the
    branch `GITHUB_HEAD_REF` sets, falling back to `GITHUB_SHA` and
    `GITHUB_REF_NAME` where either is unset or empty.
    """
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    gate = readout(file, {})

    assert (gate.commit, gate.ref) == (commit, ref)
