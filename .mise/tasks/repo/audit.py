#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "audit"
# MISE description = "Audit the declarations, task commands, programs, and wheel"
"""
Runs every check `thermur.repo.checks` declares over the checkout in the
working directory, as the `repo:audit` task, printing each finding as a
GitHub Actions annotation and exiting nonzero where any check makes one.
"""

from thermur.repo.checkout import Checkout
from thermur.repo.checks   import Check


def main():
    """
    Prints the annotation of every finding the checks make over the checkout
    in the working directory, once each where several checks read the same
    invalid document, then exits with status `1` where any check made one
    and `0` otherwise.
    """
    checkout = Checkout()
    findings = dict.fromkeys(
        finding
        for check in Check.__subclasses__()
        for finding in check(checkout=checkout).findings
    )
    for finding in findings:
        print(finding.annotation)

    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
