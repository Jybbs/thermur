"""
Pins that the cruft report runs both of knip's passes, printing their
annotations under GitHub Actions, and fails where either fails.
"""

from common.edits import executable
from common.tasks import run_site
from pathlib      import Path
from pytest       import Config, MonkeyPatch, mark


@mark.parametrize("failing", ["cycles", "unused"])
def test_the_cruft_report_runs_both_passes_and_fails_where_either_fails(
    failing      : str,
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    site         : Path
):
    """
    Asserts that the cruft report runs knip's pass over import cycles after
    a first pass that fails as well as after one that passes, and exits
    nonzero where either pass failed.
    """
    executable(
        site / "node_modules/.bin/knip",
        '#!/bin/sh\nprintf "%s\\n" "$*"\n'
        'case "$*" in *--cycles*) test "$FAILING" != cycles ;; '
        '*) test "$FAILING" != unused ;; esac\n'
    )
    monkeypatch.setenv("FAILING", failing)
    reported = run_site(site, pytestconfig, "cruft")

    assert (reported.returncode, reported.stdout.splitlines()) == (
        1,
        [
            "--reporter symbols --marker", "--cycles --reporter symbols --marker"
        ]
    )
