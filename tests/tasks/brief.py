"""
Pins what `gha:brief` writes and the verdict it exits with, meaning the
summary each workflow's template renders, the variables it reads the commit
and the branch from, where it writes the summary, and the status it returns
for the results of the jobs its gate waited on.
"""

from collections.abc  import Callable
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
) -> Callable[[str, dict], object]:
    """
    Enters the worktree root, whose `.github/scripts/summaries/` the
    task reads, and sets the variables a runner sets for a pull request's
    workflow run.

    Returns:
        A builder reading the gate of a run of the workflow file named and
        the `needs` context given, as the task reads it.
    """
    monkeypatch.chdir(pytestconfig.rootpath)
    for name, value in {
        "GITHUB_REF_NAME"   : "55/merge",
        "GITHUB_REPOSITORY" : "Jybbs/thermur",
        "GITHUB_SERVER_URL" : "https://github.com",
        "GITHUB_SHA"        : SHA
    }.items():
        monkeypatch.setenv(name, value)

    module = load_task(".mise/tasks/gha/brief.py")

    def build(file: str, needs: dict) -> object:
        """
        Reads the gate of a run of `.github/workflows/<file>` waiting on the
        jobs `needs` describes, under the name the workflow declares.
        """
        path = Path(".github/workflows", file)
        monkeypatch.setenv(
            "GITHUB_WORKFLOW",
            Workflow.read(pytestconfig.rootpath, path).name
        )
        monkeypatch.setenv(
            "GITHUB_WORKFLOW_REF",
            f"Jybbs/thermur/{path}@refs/pull/55/merge"
        )
        monkeypatch.setenv("NEEDS", dumps(needs))
        return module.Readout.running()

    return build


def job(result: str = "success", **outputs: str) -> dict:
    """
    Describes one job the gate waited on as the `needs` context holds it.
    """
    return {"outputs": outputs, "result": result}


@mark.parametrize(
    ("file", "needs"),
    [
        param("ci.yml", {"check": job(coverage="https://example.invalid/1")}, id="ci"),
        param("ci.yml", {"check": job("failure")}, id="ci-failed"),
        param(
            "draft.yml",
            {"cut": job(state="created", url="https://u/d", version="0.2.0")},
            id = "draft-created"
        ),
        param(
            "draft.yml",
            {"cut": job(state="kept", url="https://u/d", version="0.2.0")},
            id = "draft-kept"
        ),
        param(
            "draft.yml",
            {"cut": job(state="skipped", url="https://u/r", version="0.1.0")},
            id = "draft-skipped"
        ),
        param(
            "draft.yml",
            {"cut": job(state="unchanged", url="", version="0.1.0")},
            id = "draft-unchanged"
        ),
        param("draft.yml", {"cut": job("failure")}, id="draft-failed"),
        param("warm.yml", {"kit": job(), "prune": job()}, id="warm"),
        param("warm.yml", {"kit": job(), "prune": job("failure")}, id="warm-failed")
    ]
)
def test_each_workflow_renders_its_summary(
    file     : str,
    needs    : dict,
    readout  : Callable[[str, dict], object],
    snapshot : SnapshotAssertion
):
    """
    Asserts that a workflow file with a template of its own renders that
    template, naming the coverage report the coverage row attached or the
    path the draft cut took, and that any other renders `base.md.j2`.
    """
    assert readout(file, needs).summary == snapshot


@mark.parametrize(
    ("results", "status"),
    [
        param(["success", "success"], 0, id="passed"),
        param(["success", "failure"], 1, id="failed"),
        param(["success", "cancelled"], 1, id="cancelled"),
        param(["success", "skipped"], 1, id="skipped")
    ]
)
def test_the_verdict_passes_only_where_every_job_succeeded(
    readout : Callable[[str, dict], object],
    results : list[str],
    status  : int
):
    """
    Asserts that the task exits `0` only where every job the gate waited on
    succeeded, and `1` where any failed, was cancelled, or was skipped.
    """
    needs = {f"job{index}": job(result) for index, result in enumerate(results)}

    assert readout("warm.yml", needs).write() == status


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
    ("variables", "commit", "ref"),
    [
        param({}, SHA, "55/merge", id="runner"),
        param(
            {"COMMIT": "fedcba9876543210", "GITHUB_HEAD_REF": "55/ci-brief-gate"},
            "fedcba9876543210",
            "55/ci-brief-gate",
            id = "pull-request"
        ),
        param({"COMMIT": "", "GITHUB_HEAD_REF": ""}, SHA, "55/merge", id="empty")
    ]
)
def test_the_commit_and_the_branch_prefer_the_pull_request_s_head(
    commit      : str,
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

    gate = readout("warm.yml", {})

    assert (gate.commit, gate.ref) == (commit, ref)
