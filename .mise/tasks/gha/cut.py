#!/usr/bin/env python3
# MISE description = "Keep one draft release open for `[project].version`"
"""
Keeps one draft GitHub release open for the `[project].version` that
`pyproject.toml` declares, as the `gha:cut` task the draft job of `🐦‍⬛
Draft` runs on the runner's own interpreter and the `gh` it carries.
`Draft.cut` creates the draft where no release carries the version, replaces
a draft's notes with the ones GitHub generates now where only a draft
carries it, and fails where a published release does.
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


def run(*command: str, input: str | None = None) -> str:
    """
    Runs `command`, passing it `input` on stdin where one is given and
    raising `CalledProcessError` on a nonzero status.

    Returns:
        Its output, with the whitespace around it removed.
    """
    return check_output(command, input=input, text=True).strip()


@dataclass(frozen=True, kw_only=True)
class Draft:
    """
    The draft release of the version `pyproject.toml` declares.
    """

    version: str

    @cached_property
    def releases(self) -> list[dict]:
        """
        Reads every release the repository holds for the version, each as
        its `draft` flag, its `id`, and its `url`, paging through the whole
        listing.
        """
        listing = run(
            "gh",
            "api",
            "--jq",
            f".[] | select(.tag_name == {dumps(self.version)}) "
            "| {draft, id, url: .html_url}",
            "--paginate",
            "repos/{owner}/{repo}/releases"
        )
        return list(map(from_json, listing.splitlines()))

    def cut(self) -> tuple[str, str]:
        """
        Takes the path the release listing calls for, printing an error
        annotation naming `mise bump` where a published release already
        carries the version.

        Returns:
            The path as `published`, `updated`, or `created`, and the URL of
            the release it leaves.
        """
        if published := [release for release in self.releases if not release["draft"]]:
            print(
                f"::error::{self.version} is already published, so `mise bump` "
                "moves `[project].version` to the next release"
            )
            return "published", published[0]["url"]

        if self.releases:
            notes = run(
                "gh",
                "api",
                "-f",
                f"tag_name={self.version}",
                "-f",
                "target_commitish=main",
                "--jq",
                "{body}",
                "--method",
                "POST",
                "repos/{owner}/{repo}/releases/generate-notes"
            )
            return "updated", run(
                "gh",
                "api",
                "--input",
                "-",
                "--jq",
                ".html_url",
                "--method",
                "PATCH",
                f"repos/{{owner}}/{{repo}}/releases/{self.releases[0]['id']}",
                input = notes
            )

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

    def emit(self) -> int:
        """
        Writes the version, the path the cut takes, and the URL to
        `$GITHUB_OUTPUT` for the gate to read, or prints them where it is
        unset.

        Returns:
            The exit status, `1` on the published path and `0` otherwise.
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

        return int(state == "published")

    @classmethod
    def read(cls) -> Self:
        """
        Reads the version `pyproject.toml` declares.
        """
        return cls(
            version = loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
                "project"
            ]["version"]
        )


if __name__ == "__main__":
    raise SystemExit(Draft.read().emit())
