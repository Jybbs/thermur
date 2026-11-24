"""
Pins `thermur fly`, meaning the steps it computes with no step named and
with step names, what it prints, its exit on a flag past its bound, and
where the help of its steps comes from.
"""

from pytest import CaptureFixture, MonkeyPatch, fixture, mark, param, raises

from thermur.cli            import app
from thermur.pipeline.graph import Graph


@fixture
def computed(monkeypatch: MonkeyPatch) -> list[tuple[bool, int, int, tuple[str, ...]]]:
    """
    Stands in for computing the graph, recording whether each graph records
    its run, the flock size and the radio's table its settings hold, and the
    steps named, and returning each step's name as its result.

    Returns:
        The record of each call, in order.
    """
    calls: list[tuple[bool, int, int, tuple[str, ...]]] = []

    def compute(self: Graph, *steps: str) -> dict[str, object]:
        """
        Records the call and names each step as its result.
        """
        calls.append(
            (self.recorded, self.settings.plan.drones, self.settings.radio.slots, steps)
        )
        return {step: f"<{step}>" for step in steps}

    monkeypatch.setattr(Graph, "compute", compute)
    return calls


@mark.parametrize(
    ("tokens", "steps", "printed"),
    [
        param((), ("frames", "statistics"), "<statistics>\n", id="reference"),
        param(("frames", "air"), ("frames", "air"), "<frames>\n<air>\n", id="named")
    ]
)
def test_a_flight_computes_its_steps_in_a_recorded_run_and_prints_them(
    capsys   : CaptureFixture[str],
    computed : list[tuple[bool, int, int, tuple[str, ...]]],
    printed  : str,
    steps    : tuple[str, ...],
    tokens   : tuple[str, ...]
):
    """
    Asserts that `thermur fly` computes the frames and the statistics in
    a recorded run and prints the statistics, and that given step names it
    computes and prints each one, under the settings its flags set.
    """
    app(
        ["fly", *tokens, "--plan.drones", "30", "--radio.slots", "24"],
        result_action = "return_value"
    )

    assert computed == [(True, 30, 24, steps)]
    assert capsys.readouterr().out == printed


@mark.parametrize(
    "tokens",
    [
        param(("--plan.drones", "0"), id="field"),
        param(("--radio.slots", "6"), id="neighbors")
    ]
)
def test_a_flag_past_its_bound_exits_with_status_two_computing_nothing(
    computed : list[tuple[bool, int, int, tuple[str, ...]]],
    tokens   : tuple[str, ...]
):
    """
    Asserts that a flag whose value fails its own bound, or the bound the
    settings set across fields, exits with status 2 before any step is
    computed.
    """
    with raises(SystemExit) as exited:
        app(["fly", *tokens], result_action="return_value")

    assert exited.value.code == 2
    assert computed == []


@mark.parametrize(
    ("entry", "parse_docstring"),
    [
        param(None, False, id="signature"),
        param("Each step of the graph to compute.", True, id="docstring")
    ]
)
def test_the_help_of_the_steps_comes_from_the_docstring(
    entry           : str | None,
    parse_docstring : bool
):
    """
    Asserts that the steps of `thermur fly` take no help from the signature
    and take the entry of the docstring's `Args` block as written, which
    `thermur fly --help` prints beside `STEPS`.
    """
    collection = app["fly"].assemble_argument_collection(
        parse_docstring = parse_docstring
    )

    assert collection["STEPS"].parameter.help == entry
