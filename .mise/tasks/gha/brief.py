#!/usr/bin/env -S uv run --exact --locked --script
# MISE description = "Write the workflow run's summary and exit with the gate's verdict"
# /// script
# dependencies    = ["minijinja==2.24.0"]
# requires-python = ">=3.14"
# ///
"""
Writes the summary of a workflow run and exits with the verdict of every job
its gate waited on, as the `gha:brief` task the `📡 Readout` gate of each
workflow runs. `Readout` fills in the template `.github/scripts/summaries/`
holds under the running workflow file's own name, or `base.md.j2` where it
holds none, through minijinja.
"""

from contextlib  import nullcontext
from dataclasses import dataclass
from json        import loads
from minijinja   import Environment, load_from_path
from os          import environ
from pathlib     import Path
from typing      import Self


@dataclass(frozen=True, kw_only=True)
class Readout:
    """
    The `📡 Readout` gate of one workflow run, holding the `needs` context
    of the jobs it waited on beside what the runner's variables say of the
    run.
    """

    commit     : str
    file       : Path
    needs      : dict[str, dict]
    ref        : str
    repository : str
    server     : str
    workflow   : str

    @property
    def link(self) -> str:
        """
        Writes the Markdown link to the commit, shown as the seven
        characters GitHub shortens a commit to.
        """
        return (
            f"[`{self.commit[:7]}`]"
            f"({self.server}/{self.repository}/commit/{self.commit})"
        )

    @property
    def passed(self) -> bool:
        """
        Tells whether every job the gate waited on succeeded or was skipped,
        a job being skipped only where its own condition left it out or a
        job it waits for ended otherwise, which that job's result already
        reads.
        """
        return all(
            job["result"] in {"skipped", "success"} for job in self.needs.values()
        )

    @property
    def summary(self) -> str:
        """
        Fills in the template named for the workflow's file, or `base.md.j2`
        where none is, from the `needs` context, so the template reads any
        output a job the gate waited on declares, naming the workflow by the
        words its name carries after its glyph.
        """
        folder = Path(".github/scripts/summaries")
        named  = self.file.with_suffix(".md.j2").name
        return Environment(
            loader        = load_from_path(folder),
            lstrip_blocks = True,
            trim_blocks   = True
        ).render_template(
            named if (folder / named).is_file() else "base.md.j2",
            commit   = self.link,
            needs    = self.needs,
            passed   = self.passed,
            ref      = self.ref,
            workflow = self.workflow.partition(" ")[2]
        )

    @classmethod
    def running(cls) -> Self:
        """
        Reads the gate of the running workflow run from `NEEDS`, the `needs`
        context as JSON, and from the variables the runner sets, taking the
        commit from `COMMIT` where the workflow sets it to the head of a
        pull request's branch and from `GITHUB_SHA` otherwise.
        """
        return cls(
            commit     = environ.get("COMMIT") or environ["GITHUB_SHA"],
            file       = Path(environ["GITHUB_WORKFLOW_REF"].partition("@")[0]),
            needs      = loads(environ["NEEDS"]),
            ref        = environ.get("GITHUB_HEAD_REF") or environ["GITHUB_REF_NAME"],
            repository = environ["GITHUB_REPOSITORY"],
            server     = environ["GITHUB_SERVER_URL"],
            workflow   = environ["GITHUB_WORKFLOW"]
        )

    def write(self) -> int:
        """
        Appends the summary to `$GITHUB_STEP_SUMMARY`, or prints it where
        that variable is unset.

        Returns:
            The verdict as an exit status, `0` where every job succeeded or
            was skipped and `1` otherwise.
        """
        path = environ.get("GITHUB_STEP_SUMMARY")
        with open(path, "a", encoding="utf-8") if path else nullcontext() as page:
            print(self.summary, file=page)

        return 0 if self.passed else 1


if __name__ == "__main__":
    raise SystemExit(Readout.running().write())
