"""
Pins `Measure`, `Trial`, and `summarized`, meaning how a measurement scores
a candidate, which candidates a trial keeps, the composites it flies, the
table a flown trial returns, and the mean and standard error across seeds.
"""

from common.drones import short, still_air
from dataclasses   import fields
from math          import inf
from numpy         import array
from pytest        import mark, param
from xarray        import Dataset, concat

from thermur.controller.components import Constant, Elastic, Weighted
from thermur.controller.schemas    import Composite
from thermur.metrics.trials        import Measure, REFERENCE, Trial, summarized
from thermur.pipeline.settings     import Settings

TABLE = Dataset(
    {
        "spacing_m": (
            ("candidate", "condition", "seed"),
            array([[[4.0, 6.0], [1.0, 1.0]], [[2.0, 2.0], [0.0, 4.0]]])
        )
    },
    coords = {"candidate": [REFERENCE, "other"], "condition": ["dense", "sparse"]}
)


@mark.parametrize(
    ("measure", "expected"),
    [
        param(
            Measure(condition="dense", goal=inf, statistic="spacing_m"),
            [[4.0, 6.0], [2.0, 2.0]],
            id = "larger"
        ),
        param(
            Measure(condition="dense", goal=-inf, statistic="spacing_m"),
            [[-4.0, -6.0], [-2.0, -2.0]],
            id = "smaller"
        ),
        param(
            Measure(condition="sparse", goal=2.0, statistic="spacing_m"),
            [[-1.0, -1.0], [-2.0, -2.0]],
            id = "nearest"
        ),
        param(
            Measure(statistic="spacing_m"),
            [[-1.5, -2.5], [-1.0, -1.0]],
            id = "flattest"
        )
    ]
)
def test_a_measure_scores_each_candidate_and_seed_so_larger_is_favored(
    expected : list[list[float]],
    measure  : Measure
):
    """
    Asserts that a measure reads its condition, or the spread across every
    condition, and scores it so a larger score is favored, toward infinity,
    away from it, or nearest its goal.
    """
    assert measure.score(TABLE).values.tolist() == expected


def test_a_flown_trial_tables_every_flight_beside_its_ranks():
    """
    Asserts that flying a trial returns its statistics by candidate,
    condition, and seed, whether each candidate is favored, and the rank of
    each model on each candidate's replayed flight.
    """
    flown = Trial(
        candidates = {"elastic": Elastic()},
        conditions = {"small": {"drones": 10}},
        duration_s = 2.0,
        measures   = (Measure(statistic="spacing_m"),),
        slot       = "cohesion"
    ).flown(Composite.from_settings(Settings()), still_air(), short())

    assert dict(flown.sizes)["candidate"] == 2
    assert flown.spacing_m.dims == ("candidate", "condition", "seed")
    assert list(flown.model.values) == [REFERENCE, "elastic"]
    assert flown.log_likelihood.dims == ("candidate", "model")
    assert not flown.favored.sel(candidate=REFERENCE)


@mark.parametrize(
    ("measures", "favored"),
    [
        param(
            (Measure(condition="dense", goal=-inf, statistic="spacing_m"),),
            True,
            id = "better"
        ),
        param(
            (Measure(condition="dense", goal=inf, statistic="spacing_m"),),
            False,
            id = "worse"
        ),
        param(
            (
                Measure(condition="dense", goal=-inf, statistic="spacing_m"),
                Measure(condition="sparse", goal=1.0, statistic="spacing_m")
            ),
            False,
            id = "better-and-worse"
        ),
        param(
            (Measure(condition="sparse", goal=inf, statistic="spacing_m"),),
            False,
            id = "within-error"
        )
    ]
)
def test_a_candidate_is_kept_only_where_no_measure_leaves_it_worse(
    favored  : bool,
    measures : tuple[Measure, ...]
):
    """
    Asserts that a candidate is kept where one measure leaves it better than
    the reference beyond the standard error and none leaves it worse, and
    that neither the reference nor a control scoring as the candidate does
    is ever kept.
    """
    kept = trial(*measures).favored(
        concat(
            [
                TABLE,
                (
                    TABLE.sel(candidate=["other"])
                         .assign_coords(candidate=["uncoupled"])
                )
            ],
            "candidate"
        )
    )

    assert kept.sel(candidate="other") == favored
    assert not kept.sel(candidate=REFERENCE)
    assert not kept.sel(candidate="uncoupled")


def test_a_summary_reads_the_mean_across_seeds_beside_its_standard_error():
    """
    Asserts that the summary keeps every dimension but the seed, the mean of
    each pair of seeds beside half their population spread over the root of
    the count.
    """
    summary = summarized(TABLE["spacing_m"])

    assert summary["mean"].values.tolist() == [[5.0, 1.0], [2.0, 2.0]]
    assert summary["error"].values.tolist() == [
        [1 / 2**0.5, 0.0],
        [0.0, 2 / 2**0.5]
    ]


def test_a_trial_flies_the_reference_then_each_candidate_then_each_control():
    """
    Asserts that a trial lays out the reference first, each candidate in the
    slot next, and each control with its slots changed last.
    """
    composite  = Composite.from_settings(Settings())
    composites = trial().composites(composite)

    assert list(composites) == [REFERENCE, "other", "uncoupled"]
    assert composites[REFERENCE] is composite
    assert composites["other"].coupling == Weighted()
    assert composites["uncoupled"].coupling == Constant(coupling_per_s=0.0)
    assert composites["other"].cohesion == composite.cohesion


def test_each_listed_trial_opens_a_slot_of_the_composite_once():
    """
    Asserts that each trial names a slot the composite declares, no slot
    twice, and each step's name from its slot.
    """
    trials = Trial.listed()
    slots  = [trial.slot for trial in trials]

    assert len(set(slots)) == len(slots)
    assert set(slots) <= {field.name for field in fields(Composite)}
    assert [trial.name for trial in trials] == [f"{slot}_trial" for slot in slots]


def trial(*measures: Measure) -> Trial:
    """
    Builds a trial of one coupling candidate and one control, chosen by
    `measures`.
    """
    return Trial(
        candidates = {"other": Weighted()},
        conditions = {"dense": {}, "sparse": {}},
        controls   = {"uncoupled": {"coupling": Constant(coupling_per_s=0.0)}},
        measures   = measures,
        slot       = "coupling"
    )
