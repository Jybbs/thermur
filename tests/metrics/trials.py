"""
Pins `Measure`, `Trial`, and `summarized`, meaning how a measurement scores
a candidate, which candidates a trial keeps, the composites it flies, the
table a flown trial returns, and the mean and standard error across seeds.
"""

from common.drones import short, still_air
from dataclasses   import fields
from math          import inf, sqrt
from numpy         import array
from numpy.testing import assert_allclose
from pytest        import MonkeyPatch, mark, param, raises
from types         import SimpleNamespace
from xarray        import Dataset, concat

from thermur.controller.components import Constant, Cruise, Elastic, InertialSpin, Weighted
from thermur.controller.schemas    import Composite
from thermur.environment.air       import Air
from thermur.metrics.flight        import Flight
from thermur.metrics.schemas       import Plan
from thermur.metrics.trials        import Measure, POLARIZATION, REFERENCE, SIZES, Trial, summarized
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
        ),
        param(
            Measure(level=2.0, statistic="spacing_m"),
            [[-sqrt(2.5), -sqrt(8.5)], [-sqrt(2.0), -sqrt(2.0)]],
            id = "flattest-about-level"
        )
    ]
)
def test_a_measure_scores_each_candidate_and_seed_so_larger_is_favored(
    expected : list[list[float]],
    measure  : Measure
):
    """
    Asserts that a measure reads its condition, or the spread across every
    condition about its mean or its level, and scores it so a larger score
    is favored, toward infinity, away from it, or nearest its goal.
    """
    assert_allclose(measure.score(TABLE), expected)


def test_a_flown_trial_tables_every_flight_beside_its_ranks():
    """
    Asserts that flying a trial returns its statistics by candidate,
    condition, and seed, whether each candidate is favored, and the rank of
    each model on each candidate's replayed flight.
    """
    flown = Trial(
        candidates = {"elastic": Elastic()},
        conditions = {"small": {"drones": 10}},
        drones     = 10,
        duration_s = 2.0,
        measures   = (Measure(statistic="spacing_m"),),
        slot       = "cohesion"
    ).flown(still_air(), Composite.from_settings(Settings()), short())

    assert dict(flown.sizes)["candidate"] == 2
    assert flown.spacing_m.dims == ("candidate", "condition", "seed")
    assert list(flown.model.values) == [REFERENCE, "elastic"]
    assert flown.log_likelihood.dims == ("candidate", "model")
    assert not flown.favored.sel(candidate=REFERENCE)


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


@mark.parametrize(
    ("conditions", "drones"),
    [
        param({"spontaneous": {}}, 9, id="trial-size"),
        param({"small": {"drones": 10}}, 10, id="condition-size")
    ]
)
def test_a_trial_flies_its_own_flock_size_unless_its_condition_sets_one(
    conditions  : dict[str, dict[str, int | float]],
    drones      : int,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a trial flies the flock size it holds rather than the
    plan's, and the size a condition sets over both, while its short replay
    flies the trial's own size.
    """
    sizes: list[int] = []

    def flown(air: Air, composite: Composite, plan: Plan) -> SimpleNamespace:
        """
        Records the flock size of the plan, returning no statistics for a
        long flight and stopping the trial at the short replay.
        """
        sizes.append(plan.drones)
        if plan.seeds == 1:
            raise LookupError
        return SimpleNamespace(statistics=Dataset())

    monkeypatch.setattr(Flight, "flown", flown)
    with raises(LookupError):
        Trial(
            candidates = {},
            conditions = conditions,
            drones     = 9,
            measures   = (),
            slot       = "heading"
        ).flown(still_air(), Composite.from_settings(Settings()), short())

    assert sizes == [drones, 9]


def test_a_trial_flies_the_reference_then_each_candidate_then_each_control():
    """
    Asserts that a trial lays out the reference first, each candidate in the
    slot next, and each control with its slots changed last.
    """
    settings   = Settings()
    composite  = Composite.from_settings(settings)
    composites = trial().composites(composite, settings.plan.similarity)

    assert list(composites) == [REFERENCE, "other", "uncoupled"]
    assert composites[REFERENCE] is composite
    assert composites["other"].coupling == Weighted()
    assert composites["uncoupled"].coupling == Constant(coupling_per_s=0.0)
    assert composites["other"].cohesion == composite.cohesion


def test_a_trial_scales_each_candidate_and_control_as_the_reference_is_scaled():
    """
    Asserts that a trial flies each candidate and each control scaled from
    a starling flock by the plan's similarity, as `Composite.from_settings`
    scales the reference.
    """
    settings   = Settings()
    similarity = settings.plan.similarity
    composites = Trial(
        candidates = {"inertial": InertialSpin()},
        conditions = {"dense": {}},
        controls   = {"fixed": {"speed": Cruise(fixed=True)}},
        measures   = (),
        slot       = "heading"
    ).composites(Composite.from_settings(settings), similarity)

    assert composites["inertial"].heading == InertialSpin().scaled(similarity)
    assert composites["fixed"].speed == Cruise(fixed=True).scaled(similarity)


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


def test_every_listed_condition_setting_no_size_flies_the_middle_size():
    """
    Asserts that each flight whose condition sets no flock size flies the
    middle of `SIZES`, whatever the plan's own flock size.
    """
    plan = Settings().plan

    assert {
        plan.changed(drones=trial.drones).changed(**changes).drones
        for trial in Trial.listed()
        for changes in trial.conditions.values()
        if "drones" not in changes
    } == {SIZES["middle"]["drones"]}


def test_every_listed_trial_flies_the_sizes_its_tables_were_measured_at():
    """
    Asserts that every flight of every listed trial flies one of the sizes
    `SIZES` holds, whatever the plan's own flock size.
    """
    plan = Settings().plan

    assert {
        plan.changed(drones=trial.drones).changed(**changes).drones
        for trial in Trial.listed()
        for changes in trial.conditions.values()
    } <= {size["drones"] for size in SIZES.values()}


def test_the_coupling_trial_favors_a_flat_polarization_only_near_a_starling_flock():
    """
    Asserts that the coupling trial scores a flock holding a disordered
    polarization of 0.15 at every spacing below one holding a polarization
    near `POLARIZATION` that changes modestly across the spacings.
    """
    (coupling,) = (trial for trial in Trial.listed() if trial.slot == "coupling")
    (measure,)  = (
        each for each in coupling.measures if each.statistic == "polarization"
    )
    table = Dataset(
        {
            "polarization": (
                ("candidate", "condition", "seed"),
                array(
                    [
                        [[0.15], [0.15], [0.15]],
                        [[POLARIZATION + 0.02], [POLARIZATION], [POLARIZATION - 0.03]]
                    ]
                )
            )
        },
        coords = {
            "candidate" : ["disordered", "ordered"],
            "condition" : ["dense", "middle", "sparse"]
        }
    )
    disordered, ordered = measure.score(table).values.ravel()

    assert disordered < ordered


def test_the_heading_trial_favors_a_spontaneous_spectral_peak_nearest_zero():
    """
    Asserts that the heading trial scores a spontaneous correlation peaking
    at zero frequency, the overdamped form measured on starling flocks,
    above one peaking at a traveling wave's 0.45 Hz.
    """
    (heading,) = (trial for trial in Trial.listed() if trial.slot == "heading")
    (measure,) = (
        each for each in heading.measures if each.statistic == "spectral_peak_hz"
    )
    table = Dataset(
        {
            "spectral_peak_hz": (
                ("candidate", "condition", "seed"),
                array([[[0.0, 0.0]], [[0.45, 0.45]]])
            )
        },
        coords = {"candidate": ["relaxing", "waving"], "condition": ["spontaneous"]}
    )

    assert measure.score(table).values.tolist() == [[0.0, 0.0], [-0.45, -0.45]]


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
