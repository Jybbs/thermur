#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "audit"
# MISE description = "Audit the declarations, tasks, programs, wheel, and workflows"
"""
Runs every check `thermur.repo.checks` declares over the checkout in the
working directory, as the `repo:audit` task, printing each finding as
a GitHub Actions annotation and exiting nonzero where any check makes a
finding at the error level.
"""

from thermur.repo.checkout import Checkout
from thermur.repo.checks   import Check
from thermur.repo.schemas  import Level


def main():
    """
    Prints the annotation of every finding the checks make over the checkout
    in the working directory, once each where several checks read the same
    invalid document, then exits with status `1` where any finding is an
    error and `0` where every one is a warning or none is made.
    """
    checkout = Checkout()
    findings = dict.fromkeys(
        *check(checkout=checkout).findings for check in Check.__subclasses__()
    )
    for finding in findings:
        print(finding.annotation)

    raise SystemExit(any(finding.level is Level.ERROR for finding in findings))


if __name__ == "__main__":
    main()
