#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "bump"
# MISE description = "Move `[project].version` to the next release"
# USAGE flag "--major" help="Move to the next major release"
# USAGE flag "--patch" help="Move to the next patch release"
"""
Moves the `[project].version` that `pyproject.toml` declares to the next
minor release, or the next patch or major one under `--patch` or `--major`,
as the `repo:bump` task. `uv version --bump --frozen` writes the manifest
without resolving the lockfile again, and the task writes the same version
into the `thermur` entry of `uv.lock` and the `version` of `CITATION.cff`,
leaving every other pin where it stands. mise passes each flag the usage
spec above declares through a `usage_` variable.
"""

from os      import environ
from pathlib import Path
from sys     import stderr

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Edit, Lock


def main():
    """
    Moves the version and prints the one it moved to, exiting with status
    `2` where `--patch` and `--major` are both given.
    """
    named = {part: environ.get(f"usage_{part}") == "true" for part in (
        "major", "patch"
    )}
    if all(named.values()):
        print(
            "`--patch` and `--major` name two releases, so give one at most",
            file = stderr
        )
        raise SystemExit(2)

    checkout = Checkout()
    current  = checkout.manifest.version
    version  = checkout.run(
        "uv",
        "version",
        "--bump",
        next((part for part, given in named.items() if given), "minor"),
        "--frozen",
        "--short",
        capture = True
    ).stdout.strip()
    for edit in [
        Edit(
            ahead   = "^version: ",
            current = current,
            file    = checkout.citation.file,
            newest  = version
        ),
        Edit(
            ahead   = r'^name = "thermur"\nversion = "',
            current = current,
            file    = Path(Lock.model_config["title"]),
            newest  = version
        )
    ]:
        edit.write(checkout.root)

    print(version)


if __name__ == "__main__":
    main()
