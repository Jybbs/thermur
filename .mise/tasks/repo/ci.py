#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "ci"
# MISE description = "Run each row's task once `mise doctor project` passes"
"""
Runs `mise doctor project` and then every task the rows of the pull-request
workflows under `.github/workflows/` run, as the `repo:ci` task, so a laptop
runs what each row of `🔥 CI` runs. The tasks run under `--force`, because
the run establishes a verdict, and a task whose `sources` look current would
otherwise pass without running.
"""

from subprocess import run

from thermur.repo.checkout import Checkout


def main():
    """
    Exits with the status `mise doctor project` exits with where it fails,
    and otherwise with the status of one `mise run` of every row's task,
    each separated from the next by `:::`.
    """
    if doctor := run(["mise", "doctor", "project"]).returncode:
        raise SystemExit(doctor)

    tasks = " ::: ".join(Checkout().rows).split()
    raise SystemExit(run(["mise", "run", "--force", *tasks]).returncode)


if __name__ == "__main__":
    main()
