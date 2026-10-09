"""
Pins that mise puts each program under `.mise/bin/` on the path of a command
it runs, and that each passes every argument it receives to `uv run --exact
--locked` on the program of its name in the `.venv` of the checkout holding
it, whatever folder it runs from.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from pathlib          import Path
from pytest           import Config, mark
from subprocess       import run

from thermur.repo.checkout import Checkout

PROGRAMS = [path.name for path in Checkout(root=Path(__file__).parents[2]).scripts]


@mark.parametrize("program", PROGRAMS)
def test_each_program_passes_every_argument_to_a_locked_run(
    program      : str,
    pytestconfig : Config,
    stand_in     : Callable[[str, str], StandIn],
    tmp_path     : Path
):
    """
    Asserts that the program named `program`, started from a folder
    outside the checkout, runs `uv run --exact --locked` on the program
    of the same name in the `.venv` of the checkout holding it, followed
    by each argument as it was given, a word holding a space included.
    The paths compare once resolved, since the script names the root as
    `.mise/bin/../..`.
    """
    root = pytestconfig.rootpath
    stand_in("uv", 'printf "%s\\n" "$@"')

    started = run(
        [root / ".mise/bin" / program, "--version", "two words"],
        capture_output = True,
        check          = True,
        cwd            = tmp_path,
        text           = True
    ).stdout.splitlines()

    assert started[:4] == ["run", "--exact", "--locked", "--project"]
    assert [Path(word).resolve() for word in started[4:6]] == [
        root,
        root / ".venv/bin" / program
    ]
    assert started[6:] == ["--version", "two words"]


@mark.parametrize("program", PROGRAMS)
def test_mise_puts_each_program_on_the_path_of_a_command_it_runs(
    program      : str,
    pytestconfig : Config
):
    """
    Asserts that a command `mise x` runs from a folder below the root finds
    `program` at `.mise/bin/<program>`, which `_.path` names relative to the
    checkout rather than to the working directory, ahead of any program of
    that name the machine already holds.
    """
    root  = pytestconfig.rootpath
    found = run(
        ["mise", "x", "--", "sh", "-c", f"command -v {program}"],
        capture_output = True,
        check          = True,
        cwd            = root / "tests",
        text           = True
    )

    assert found.stdout.strip() == str(root / ".mise/bin" / program)
