"""
Pins `Flight`, meaning that a flight repeats from its seed, that its batch
flies each seed as one flight alone would, the tables a recorded run saves,
the first drone's turn, the replay ranking composites, and the helpers each
statistic reads.
"""

from common.drones import short, still_air
from dataclasses   import replace
from jax           import jit
from jax.numpy     import array
from jax.random    import key, split
from numpy         import (
    allclose,
    arange,
    exp,
    full,
    isfinite,
    isnan,
    linalg,
    linspace,
    meshgrid,
    ones,
    pi,
    stack,
    where
)
from numpy.testing import assert_allclose, assert_array_equal
from pytest        import fixture, mark, param

from thermur.controller.components import Consensus, Overdamped
from thermur.controller.schemas    import Composite, Heading
from thermur.metrics.flight        import Flight, first_zero, kept, ratio_on_hull, response
from thermur.pipeline.settings     import Settings

AIR = still_air()


@fixture(scope="module")
def composite() -> Composite:
    """
    Reads the default composite from the settings.
    """
    return Composite.from_settings(Settings())


@fixture(scope="module")
def flight(composite: Composite) -> Flight:
    """
    Flies the short plan under the default composite through still air.
    """
    return Flight.flown(composite, AIR, short())


def test_a_flight_repeats_from_its_seed_and_differs_across_seeds(
    composite : Composite,
    flight    : Flight
):
    """
    Asserts that flying one plan again keeps every frame, that each seed
    of the batch flies a flock of its own, and that another `seed` flies
    another batch.
    """
    again   = Flight.flown(composite, AIR, short())
    another = Flight.flown(composite, AIR, short(seed=1))

    assert_array_equal(again.frames.positions, flight.frames.positions)
    assert (flight.frames.positions[0] != flight.frames.positions[1]).any()
    assert (another.frames.positions != flight.frames.positions).any()


def test_a_lattice_s_border_is_as_dense_as_its_inside():
    """
    Asserts that a regular lattice, every drone a step from its nearest
    neighbor, reads a border density equal to its inside's, and a cloud
    whose inside sits at half the step reads its border sparser.
    """
    lattice = stack(meshgrid(*[linspace(0.0, 4.0, 5)] * 3), -1).reshape(-1, 3)
    inside  = (abs(lattice - 2.0) < 2.0).all(-1)

    assert_allclose(ratio_on_hull(lattice, ones(len(lattice))), 1.0)
    assert ratio_on_hull(lattice, where(inside, 0.5, 1.0)) < 1.0


@mark.parametrize(
    ("heading", "held"),
    [param(Overdamped(), True, id="confined"), param(Consensus(), False, id="free")]
)
def test_a_heading_holding_the_speed_keeps_every_drone_at_the_cruise(
    composite : Composite,
    heading   : Heading,
    held      : bool
):
    """
    Asserts that every drone flies every kept frame at `cruise_m_s` under a
    heading holding the speed, and that the speeds spread under one leaving
    it free.
    """
    speeds = linalg.norm(
        Flight.flown(
            replace(composite, heading=heading),
            AIR,
            short()
        ).frames.velocities,
        axis = -1
    )

    assert allclose(speeds, 10.0, rtol=1e-4) == held


def test_a_replay_ranks_every_composite_it_is_given(
    composite : Composite,
    flight    : Flight
):
    """
    Asserts that replaying the frames gives one drift a frame but the last,
    and one log-likelihood and one coefficient for each composite named.
    """
    ranked = flight.likelihoods({"one": composite, "two": composite}, AIR)

    assert flight.drifts(composite, AIR).shape == (2, 19, 12, 3)
    assert list(ranked.model.values) == ["one", "two"]
    assert isfinite(ranked.log_likelihood).all()
    assert_allclose(ranked.log_likelihood[0], ranked.log_likelihood[1])


def test_a_response_spreading_at_a_constant_speed_reads_that_speed():
    """
    Asserts that shells of drones turning at times proportional to their
    distance from the first drone read the speed of that spread, and a reach
    of one where the farthest shell ends the flight turned through the whole
    turn.
    """
    plan      = short(turn_deg=90.0, turning_window_s=0.1)
    distances = array([0.0, 10.0, 10.0, 30.0, 30.0, 50.0, 50.0, 70.0, 70.0, 90.0, 90.0])

    speed, reach = response(
        distances = distances,
        plan      = plan,
        rotation  = exp(
            1j * pi / 2 * (arange(
                60
            )[:, None] >= (distances / 20.0 / plan.frame_s)[None])
        )
    )

    assert_allclose((speed, reach), (20.0, 1.0), rtol=1e-6)


def test_each_seed_of_the_batch_flies_as_it_does_alone(
    composite : Composite,
    flight    : Flight
):
    """
    Asserts that the frames `vmap` gives each seed match the frames one
    flight of that seed's key gives with no batch.
    """
    plan = short()

    alone = jit(kept, static_argnames="plan")(
        air       = AIR,
        composite = composite,
        key       = split(key(plan.seed), plan.seeds)[1],
        plan      = plan
    )

    assert_allclose(flight.frames.positions[1], alone.positions, atol=1e-3)
    assert_allclose(flight.frames.velocities[1], alone.velocities, atol=1e-4)


@mark.parametrize(
    ("curve", "expected"),
    [
        param([1.0, 0.5, -0.1, 0.3], 2, id="crossing"),
        param([1.0, 0.0], 1, id="zero"),
        param([1.0, 0.5], 2, id="none")
    ]
)
def test_the_first_zero_is_the_first_value_at_or_below_zero(
    curve    : list[float],
    expected : int
):
    """
    Asserts that `first_zero` finds the first value at or below zero, or the
    curve's length where none is.
    """
    assert first_zero(array(curve)) == expected


def test_the_first_drone_holds_its_turn_and_the_turn_reaches_the_flock(
    composite: Composite
):
    """
    Asserts that from `turn_at_s` on, the first drone flies the turned
    velocity, and the turn's speed and reach read as finite values.
    """
    plan  = short(duration_s=4.0, turn_at_s=1.0, turn_deg=90.0)
    flown = Flight.flown(composite, AIR, plan)

    speed, reach = flown.turn_response

    assert_allclose(
        flown.frames.velocities[:, 10:, 0],
        full((2, 30, 3), plan.turned),
        atol = 1e-5
    )
    assert isfinite(speed).all()
    assert (speed >= 0).all()
    assert isfinite(reach).all()


def test_the_frames_table_holds_each_kept_frame_at_its_time(flight: Flight):
    """
    Asserts that the frames table lays out the positions and velocities by
    seed, time, drone, and axis, each frame at the end of its steps.
    """
    table = flight.dataset

    assert table.positions.dims == ("seed", "time", "drone", "axis")
    assert table.positions.shape == (2, 20, 12, 3)
    assert_allclose(table.time, 0.1 * arange(1, 21))


def test_the_statistics_table_holds_each_statistic_by_seed(flight: Flight):
    """
    Asserts that the statistics table holds one value per seed of each
    statistic, the alignments by distance bin and by rank, the polarization
    between zero and one, and no turn's response where the plan sets none.
    """
    table = flight.statistics

    assert set(table.data_vars) == {
        "alignment_by_distance", "alignment_by_rank", "aspect_ratio",
        "autocorrelation_s", "border_density", "fragments",
        "peak_height", "peak_m", "polarization",
        "spacing_m", "spectral_peak_hz", "turn_rate_per_s",
        "turn_reach", "turn_speed_m_s"
    }
    assert dict(table.sizes) == {"distance": 32, "rank": 7, "seed": 2}
    assert_allclose(table.distance[[0, -1]], (1 / 16, 4 - 1 / 16))
    assert ((table.polarization > 0) & (table.polarization <= 1)).all()
    assert isnan(table.turn_reach).all()
    assert isnan(table.turn_speed_m_s).all()
    assert (table.fragments >= 1).all()
