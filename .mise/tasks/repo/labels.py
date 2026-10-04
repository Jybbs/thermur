#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "labels"
# MISE confirm     = "Write every label `.github/labels.toml` declares to GitHub?"
# MISE description = "Write each registry label to GitHub and list those it omits"
"""
Writes every label `.github/labels.toml` declares to the repository the
checkout's remote names, as the `repo:labels` task, once mise's prompt is
answered, then prints each live label the registry omits, which stays on
GitHub until a rename or a deletion by hand settles it.
"""

from subprocess import check_output, run

from thermur.repo.checkout import Checkout


def main():
    """
    Creates or updates each label the registry declares through `gh label
    create --force`, stopping on the first that `gh` fails to write, then
    reads every label on the repository through `gh api --paginate` and
    prints each one the registry omits beneath a line naming what they are.
    """
    labels = Checkout().labels
    for label in labels.rows:
        run(label.command, check=True)

    live = check_output(
        ["gh", "api", "--jq", ".[].name", "--paginate", "repos/{owner}/{repo}/labels"],
        text = True
    ).splitlines()
    if omitted := labels.omits(live):
        print(
            f"`{labels.file}` omits these live labels, left for a rename or "
            "deletion by hand:",
            *omitted,
            sep = "\n"
        )


if __name__ == "__main__":
    main()
