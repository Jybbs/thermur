#!/usr/bin/env python3
# MISE description = "Draft the release of `[project].version` where a push moved it"
"""
Drafts the GitHub release of the `[project].version` that `pyproject.toml`
declares, as the `gha:cut` task the draft job of `🐦‍⬛ Draft` runs on the
runner's own interpreter and the `gh` it carries. `Draft.cut` cuts nothing
where a push left the version where it stood, creates the draft where
no release carries the version, leaves a draft that already carries it
untouched, and skips the cut with a warning where a published release does.
"""

from contextlib  import nullcontext
from dataclasses import dataclass
from functools   import cached_property
from json        import dumps, loads as from_json
from os          import environ
from pathlib     import Path
from subprocess  import check_output
from tomllib     import loads
from typing      import Self


def run(*command: str) -> str:
    """
    Runs `command`, raising `CalledProcessError` on a nonzero status.

    Returns:
        Its output, with the whitespace around it removed.
    """
    return check_output(command, text=True).strip()


@dataclass(frozen=True, kw_only=True)
class Draft:
    """
    The draft release of the version `pyproject.toml` declares, beside the
    version the commit before declared where a push to `main` runs the cut.
    """

    previous : str | None
    version  : str

    @cached_property
    def releases(self) -> list[dict]:
        """
        Reads every release the repository holds for the version, each as
        its `draft` flag and its `url`, paging through the whole listing.
        """
        listing = run(
            "gh",
            "api",
            "--jq",
            f".[] | select(.tag_name == {dumps(self.version)}) "
            "| {draft, url: .html_url}",
            "--paginate",
            "repos/{owner}/{repo}/releases"
        )
        return list(map(from_json, listing.splitlines()))

    def cut(self) -> tuple[str, str]:
        """
        Takes the path the release listing calls for and prints a warning
        where a published release already carries the version.

        Returns:
            The path as `unchanged`, `skipped`, `kept`, or `created`, and the
            URL of the release it leaves, or nothing where it reads none.
        """
        if self.previous == self.version:
            return "unchanged", ""

        if published := [release for release in self.releases if not release["draft"]]:
            print(
                f"::warning::{self.version} is already published, so the cut is "
                "skipped until `[project].version` moves to the next release"
            )
            return "skipped", published[0]["url"]

        if self.releases:
            return "kept", self.releases[0]["url"]

        return "created", run(
            "gh",
            "release",
            "create",
            "--draft",
            "--generate-notes",
            "--target",
            "main",
            self.version
        )

    @staticmethod
    def declared(manifest: str) -> str:
        """
        Reads the `[project].version` the text of a `pyproject.toml`
        declares.
        """
        return loads(manifest)["project"]["version"]

    def emit(self):
        """
        Writes the version, the path the cut takes, and the URL to
        `$GITHUB_OUTPUT` for the gate to read, or prints them where it is
        unset.
        """
        state, url = self.cut()
        path       = environ.get("GITHUB_OUTPUT")
        with open(path, "a", encoding="utf-8") if path else nullcontext() as output:
            print(
                f"state={state}",
                f"url={url}",
                f"version={self.version}",
                file = output,
                sep  = "\n"
            )

    @classmethod
    def read(cls) -> Self:
        """
        Reads the version `pyproject.toml` declares, and for a push the
        version it declared at `HEAD~1`, which a `workflow_dispatch` run
        leaves unread so the cut runs whatever the version is.
        """
        return cls(
            previous = cls.declared(run("git", "show", "HEAD~1:pyproject.toml"))
            if environ.get("GITHUB_EVENT_NAME") == "push"
            else None,
            version = cls.declared(Path("pyproject.toml").read_text(encoding="utf-8"))
        )


if __name__ == "__main__":
    Draft.read().emit()
