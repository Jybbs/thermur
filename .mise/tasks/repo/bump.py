#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "bump"
# MISE description = "Move `[project].version` to the next release"
# USAGE flag "--major" help="Move to the next major release"
# USAGE flag "--patch" help="Move to the next patch release"
"""
Moves the `[project].version` that `pyproject.toml` declares to the next
minor release, or the next patch or major one under `--patch` or `--major`,
as the `repo:bump` task. `uv version --bump --frozen` writes the manifest
without resolving the lockfile again, after the task has written the release
`--dry-run` prints into the `version` of `CITATION.cff` and the `thermur`
entry of `uv.lock`, so a citation naming another release fails the bump
before any file moves, and every other pin stays where it stands. mise
passes each flag the usage spec above declares through a `usage_` variable.
"""

from os  import environ
from sys import stderr

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Edit


def main():
    """
    Moves the version and prints the one it moved to, exiting with status
    `2` where `--patch` and `--major` are both given.
    """
    named = [
        part for part in ("major", "patch") if environ.get(f"usage_{part}") == "true"
    ]
    if len(named) > 1:
        print(
            "`--patch` and `--major` name two releases, so give one at most",
            file = stderr
        )
        raise SystemExit(2)

    checkout = Checkout()
    current  = checkout.manifest.version
    bump     = (
        "uv", "version", "--bump",
        next(iter(named), "minor"),
        "--frozen", "--short"
    )
    version = checkout.run(*bump, "--dry-run", capture=True).stdout.strip()
    for ahead, file in [
        ("^version: ", checkout.citation.file),
        (r'^name = "thermur"\nversion = "', checkout.lock.file)
    ]:
        Edit(
            ahead   = ahead,
            current = current,
            file    = file,
            newest  = version
        ).write(checkout.root)

    checkout.run(*bump, capture=True)
    print(version)


if __name__ == "__main__":
    main()
