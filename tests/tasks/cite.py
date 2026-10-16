"""
Pins what `docs:cite` prints and exits with, meaning the definition of each
study named in the order given, one line per key naming no entry, and the
argument mise holds a call to.
"""

from collections.abc import Callable
from pytest          import CaptureFixture, Config, MonkeyPatch, raises
from subprocess      import run
from types           import ModuleType

from thermur.repo.checkout import Checkout


def test_a_call_naming_no_key_is_refused(pytestconfig: Config):
    """
    Asserts that the task declares its keys as a required argument, so mise
    refuses a call naming none before the script runs.
    """
    called = run(
        ["mise", "run", "docs:cite"],
        capture_output = True,
        cwd            = pytestconfig.rootpath,
        text           = True
    )

    assert called.returncode
    assert "Missing required arg: <key>" in called.stderr


def test_a_key_naming_no_entry_prints_no_definition(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a key no entry holds is named on standard error and the
    task exits `1` without printing the definition of the key that does name
    an entry.
    """
    monkeypatch.chdir(checkout.root)
    monkeypatch.setattr("sys.argv", ["cite", "storms2019", "smith2020"])

    with raises(SystemExit) as exited:
        load_task(".mise/tasks/docs/cite.py").main()

    assert exited.value.code == 1
    assert capsys.readouterr() == (
        "", "`smith2020` names no entry in `docs/references.bib`\n"
    )


def test_each_definition_is_printed_in_the_order_given(
    capsys      : CaptureFixture[str],
    checkout    : Checkout,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the task prints the definition of each key in the order
    given, with a blank line between two of them.
    """
    monkeypatch.chdir(checkout.root)
    monkeypatch.setattr("sys.argv", ["cite", "storms2019", "attanasi2014"])

    load_task(".mise/tasks/docs/cite.py").main()

    assert capsys.readouterr().out == (
        f"{checkout.bibliography.cite('storms2019', 'attanasi2014')}\n"
    )
