"""
Pins `Flight`, meaning that a flight repeats from its seed, that its batch
flies each seed as one flight alone would, the tables a recorded run saves,
the first drone's turn, the replay ranking composites, and the helpers each
statistic reads.
"""

from common.drones import flock_of, short, still_air
from dataclasses   import replace
from jax           import jit
from jax.numpy     import array
from jax.random    import key, split
from numpy         import (
    allclose,
    arange,
    asarray,
    broadcast_to,
    clip,
    concat,
    cos,
    deg2rad,
    diag,
    diff,
    exp,
    full,
    isfinite,
    isnan,
    linalg,
    linspace,
    log,
    meshgrid,
    ones,
    pi,
    sin,
    stack,
    trapezoid,
    where,
    zeros
)
from numpy.testing import assert_allclose, assert_array_equal
from pytest        import FixtureRequest, fixture, mark, param

from thermur.controller.components import (
    Centroid,
    Consensus,
    Constant,
    Delayed,
    HardSphere,
    Overdamped,
    White
)
from thermur.controller.schemas import Composite, Heading, Radio, across
from thermur.environment.plant  import Plant
from thermur.environment.wrf    import WrfOutput
from thermur.metrics.flight     import Flight, first_zero, kept, ratio_on_hull, response
from thermur.pipeline.settings  import Settings

AIR = still_air()

STILL = Composite(
    cohesion = Centroid(stiffness_per_s2=0.0),
    contact  = HardSphere(weight_m_s2=0.0),
    coupling = Constant(coupling_per_s=0.0),
    heading  = Overdamped(),
    noise    = White(),
    plant    = Plant(),
    radio    = Radio(delay_s=0.0, loss=0.0),
    roost    = Centroid(stiffness_per_s2=0.0),
    speed    = Centroid(stiffness_per_s2=0.0)
)
"""
The composite whose command rate is zero on every flock, every drone hearing
each other at once.
"""


@fixture(scope="module")
def composite() -> Composite:
    """
    Reads the default composite from the settings.
    """
    return Composite.from_settings(Settings())


@fixture(
    params = [param(120.0, id="two-minutes"), param(300.0, id="five-minutes")],
    scope  = "module"
)
def default(composite: Composite, request: FixtureRequest) -> Flight:
    """
    Flies the default plan under the default composite through the
    parametric threat's air, for the plan's two minutes and for the five the
    rules page reads.
    """
    settings = Settings()
    return Flight.flown(
        composite,
        WrfOutput.from_settings(settings).air,
        settings.plan.model_copy(update={"duration_s": request.param})
    )


@fixture(scope="module")
def flight(composite: Composite) -> Flight:
    """
    Flies the short plan under the default composite through still air.
    """
    return Flight.flown(composite, AIR, short())


def laid(
    positions  : object,
    velocities : object,
    commands   : object = None,
    **changes  : object
) -> Flight:
    """
    Lays a flight out by hand under the default composite, broadcasting
    `positions`, `velocities`, and `commands`, each per drone or per frame
    and drone, across every seed and frame of the short plan carrying
    `changes`, each drone commanding its velocity where `commands` is
    `None`.
    """
    plan  = short(**changes)
    shape = (plan.seeds, plan.frames, asarray(positions).shape[-2], 3)
    return Flight(
        composite = Composite.from_settings(Settings()),
        frames    = flock_of(
            broadcast_to(positions, shape),
            broadcast_to(velocities, shape),
            None if commands is None else broadcast_to(commands, shape)
        ),
        plan = plan
    )


def test_a_cloud_on_its_axes_reads_its_longest_axis_over_its_thinnest():
    """
    Asserts that drones in pairs on each axis at 30, 20, and 10 m from the
    center read an aspect ratio of three, the longest axis over the thinnest
    rather than over the middle one.
    """
    spots = concat([diag([30.0, 20.0, 10.0]), -diag([30.0, 20.0, 10.0])] * 2)

    assert_allclose(laid(spots, [10.0, 0.0, 0.0]).aspect_ratio, 3.0, rtol=1e-5)


@mark.parametrize(
    ("rate_deg_s", "starts"),
    [param(1.0, 0.0, id="below-the-rate"), param(4.0, 1.0, id="past-the-rate")]
)
def test_a_flock_whose_heading_turns_partway_reads_one_start_a_second(
    rate_deg_s : float,
    starts     : float
):
    """
    Asserts that a flock flying straight, then turning after 1.3 s at
    `rate_deg_s`, reads one start of a spontaneous turn over the second
    of settled frames where the rate passes `turning_deg_s` of 2, and none
    where it stays below.
    """
    turned = deg2rad(rate_deg_s) * 0.1 * clip(arange(20) - 12, 0, None)
    flight = laid(
        positions  = stack([10.0 * arange(12), zeros(12), zeros(12)], -1),
        velocities = 10.0 * stack([cos(turned), sin(turned), zeros(20)], -1)[:, None]
    )

    assert_allclose(flight.turn_rate_per_s, starts)


@mark.parametrize("steps", [1, 2])
def test_a_delayed_heading_flies_otherwise_than_one_a_step_shorter(
    composite : Composite,
    steps     : int
):
    """
    Asserts that the delayed heading at a lag of `steps` integrator steps
    reads each neighbor's velocity that many steps old, so its flight
    leaves the one flown from the same seed at a lag one step shorter, the
    overdamped heading standing for a lag of none.

    Each flight couples at the starling's rate rather than the slower one
    the launch scales it to, so the step of lag between them shows within
    the short flight.
    """
    plan     = short()
    shorter  = Delayed(lag_s=(steps - 1) * plan.step_s) if steps > 1 else Overdamped()
    starling = replace(composite, coupling=Constant())

    assert not allclose(
        Flight.flown(
            replace(starling, heading=Delayed(lag_s=steps * plan.step_s)),
            AIR,
            plan
        ).frames.velocities,
        Flight.flown(replace(starling, heading=shorter), AIR, plan).frames.velocities,
        atol = 1e-3
    )


@mark.slow
def test_a_default_flight_holds_the_roost_s_height(default: Flight):
    """
    Asserts that the default flight holds its drones within 10 m of the
    roost's height on the mean over its settled frames and its seeds, where
    a plant bounding a descent tighter than a climb lifts it some 60 m above
    it in two minutes.
    """
    assert_allclose(
        asarray(default.settled.positions)[..., 2].mean(),
        Settings().roost.height_m,
        atol = 10.0
    )


@mark.slow
def test_a_default_flight_keeps_a_starling_flock_s_shape(default: Flight):
    """
    Asserts that the largest group of the default flight, averaged over
    the frames 30 s apart through two minutes and through five and over the
    seeds, stands within the span of the starling flocks the starling model
    is checked against, its middle axis 1.90 to 3.58 times its thinnest and
    its longest 4.02 to 6.94 times.

    The mean over time and seeds is what the span bounds, since the model's
    flocks report their shape as a mean over time, and one frame's shape
    stretches and folds as the flock turns at the roost's rim.
    """
    _, middle, longest = default.group_aspect_ratios.mean(0)

    assert 1.90 <= middle <= 3.58
    assert 4.02 <= longest <= 6.94


@mark.slow
def test_a_default_flight_sets_off_zigs(default: Flight):
    """
    Asserts that some drone of the default flight after two minutes and
    after five holds a zig's countdown at some kept frame, so the roost's
    turn and the neighbors' zigs reach the noise slot's state in flight.
    """
    assert (asarray(default.frames.forces)[..., 0] > 0).any()


@mark.slow
def test_a_default_flight_keeps_the_outdoor_drone_flock_s_spacing(default: Flight):
    """
    Asserts that the default flight after two minutes and after five keeps
    its drones 12 to 30 m from their nearest neighbors on the mean, the
    spacing the outdoor drone flock kept.
    """
    assert 12.0 <= float(default.spacing_m.mean()) <= 30.0


@mark.slow
def test_a_default_flight_keeps_every_drone_over_the_roost(default: Flight):
    """
    Asserts that every drone of the default flight after two minutes and
    after five sits within the roost's radius of its center and above the
    ground.
    """
    roost = Settings().roost
    spots = asarray(default.frames.positions)[:, -1]

    assert (
        linalg.norm(spots[..., :2] - asarray(roost.center_m), axis=-1) <= roost.radius_m
    ).all()
    assert (spots[..., 2] >= 0).all()


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


def test_a_heading_swinging_back_and_forth_integrates_to_its_first_zero():
    """
    Asserts that half the drones swinging their heading one way and half
    the other, over a period of ten frames, integrate their autocorrelation
    only up to the first lag at which it falls to zero, each lag's product
    averaged over the frames it spans.
    """
    swing  = cos(2 * pi * arange(20) / 10)[:, None] * where(arange(12) < 6, 1.0, -1.0)
    flight = laid(
        positions  = stack([10.0 * arange(12), zeros(12), zeros(12)], -1),
        velocities = stack([full(swing.shape, 10.0), swing, zeros(swing.shape)], -1)
    )
    headings = asarray(flight.settled.headings)[0]
    relative = headings - headings.mean(1, keepdims=True)
    curve    = [
        (relative[lag:] * relative[: 15 - lag]).sum((1, 2)).mean()
        for lag in range(15)
    ]
    curve = asarray(curve) / curve[0]

    assert_allclose(
        flight.autocorrelation_s,
        trapezoid(curve[: first_zero(curve)], dx=0.1),
        rtol = 1e-5
    )
    assert first_zero(curve) == 3


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


def test_a_long_box_reads_its_border_denser_by_the_cube_of_the_spacings():
    """
    Asserts that drones on the corners of a box 100 m long and 10 m across,
    each 10 m from its nearest corner, around four drones on its axis 20 m
    apart, read a border density eight times the inside's from each drone's
    nearest neighbor rather than its farthest.
    """
    corners = stack(meshgrid([0.0, 100.0], [0.0, 10.0], [0.0, 10.0]), -1).reshape(-1, 3)
    axis    = stack([20.0 * arange(1, 5), full(4, 5.0), full(4, 5.0)], -1)

    assert_allclose(
        laid(concat([corners, axis]), [10.0, 0.0, 0.0]).border_density,
        8.0,
        rtol = 1e-5
    )


@mark.parametrize(
    ("delay_s", "lag_s"),
    [
        param(0.0, 0.2, id="lag"),
        param(0.2, 0.0, id="delay"),
        param(0.1, 0.1, id="both")
    ]
)
def test_a_replay_reads_each_neighbor_the_radio_s_delay_and_the_lag_back(
    delay_s : float,
    lag_s   : float
):
    """
    Asserts that replaying a flock whose drones all gain one meter per
    second across the course each frame, through the delayed heading under
    a radio delay and a lag summing to two frames, turns each drone toward
    the velocity two frames back, the first frames reading the first, while
    the overdamped heading with no delay, every neighbor flying as the drone
    does, turns none.
    """
    velocities = stack([full(20, 10.0), arange(20.0), zeros(20)], -1)
    flight     = laid(
        positions  = stack([zeros(12), 10.0 * arange(12), zeros(12)], -1),
        velocities = velocities[:, None]
    )
    delayed = replace(
        STILL,
        coupling = Constant(),
        heading  = Delayed(lag_s=lag_s) if lag_s else Overdamped(),
        radio    = Radio(delay_s=delay_s, loss=0.0, range_m=1e4)
    )
    behind = velocities[clip(arange(19) - 2, 0, None)] - velocities[:-1]

    assert_allclose(
        flight.drifts(replace(STILL, coupling=Constant()), AIR),
        0.0,
        atol = 1e-5
    )
    assert_allclose(
        flight.drifts(delayed, AIR),
        broadcast_to(
            asarray(
                across(7 * Constant().coupling_per_s * behind, velocities[:-1])
            )[None, :, None],
            (2, 19, 12, 3)
        ),
        atol = 1e-4
    )


@mark.parametrize(
    ("gap_m", "groups"),
    [param(44.0, 2.0, id="inside-the-hop"), param(46.0, 12.0, id="past-the-hop")]
)
def test_two_strings_of_drones_read_their_spacing_and_their_groups(
    gap_m  : float,
    groups : float
):
    """
    Asserts that two strings of six drones 200 m apart, each drone `gap_m`
    from the next, read that nearest-neighbor distance, and two groups where
    the gap sits inside the hop of three launch spacings, or one group per
    drone where it sits past it.
    """
    string = stack([gap_m * arange(6), zeros(6), zeros(6)], -1)
    flight = laid(concat([string, string + [0.0, 200.0, 0.0]]), [10.0, 0.0, 0.0])

    assert_allclose(flight.spacing_m, gap_m, rtol=1e-6)
    assert_allclose(flight.fragments, groups)


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


def test_a_replay_reads_the_likelihood_of_its_residual_across_each_command():
    """
    Asserts that a flight whose drones command a swing of one meter per
    second across the course and back each frame, replayed through a
    composite whose drift is zero, reads the log-likelihood of a Gaussian
    in the plane across each command, its variance per axis half the mean
    squared residual, whatever the velocities the drones reach.
    """
    swing  = 10.0 * stack([ones(20), arange(20) % 2 / 10.0, zeros(20)], -1)[:, None]
    flight = laid(
        positions  = stack([10.0 * arange(12), zeros(12), zeros(12)], -1),
        commands   = swing,
        velocities = [10.0, 0.0, 0.0]
    )
    rates = asarray(
        across(
            diff(flight.frames.commands, axis=1) / 0.1,
            flight.frames.commands[:, :-1]
        )
    )

    assert_allclose(
        flight.likelihoods({"still": STILL}, AIR).log_likelihood,
        [-(log(pi * (rates**2).sum(-1).mean()) + 1)],
        rtol = 1e-5
    )


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
    Asserts that every drone commands `cruise_m_s` at every kept frame under
    a heading holding the speed, and that the commanded speeds spread under
    one leaving it free.
    """
    speeds = linalg.norm(
        Flight.flown(
            replace(composite, heading=heading),
            AIR,
            short()
        ).frames.commands,
        axis = -1
    )

    assert allclose(speeds, 10.0, rtol=1e-4) == held


def test_a_traveling_wave_reads_its_frequency_at_the_spectral_peak():
    """
    Asserts that a line of drones whose velocities across the course carry
    a wave at the flock's longest resolved wavelength, traveling at four
    thirds of a hertz, reads its spectral peak at that frequency, since the
    longest resolved wavelength leaves out a wave twice as fine and twice as
    strong traveling at 2 Hz.
    """
    spots  = stack([10.0 * arange(12), zeros(12), zeros(12)], -1)
    along  = spots[:, 0] - spots[:, 0].mean()
    times  = 0.1 * arange(20)[:, None]
    longer = cos(2 * pi * (along / (4 * along.std()) - 4 / 3 * times))
    swing  = longer + 2 * cos(2 * pi * (along / (2 * along.std()) - 2 * times))
    flight = laid(
        positions  = spots,
        velocities = stack([full(swing.shape, 10.0), swing, zeros(swing.shape)], -1)
    )

    assert_allclose(flight.spectral_peak_hz, 4 / 3, rtol=1e-5)


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


def test_lines_of_three_drones_read_their_first_g_r_peak_over_shell_volume():
    """
    Asserts that four lines of three drones 10 m apart, each line far from
    the others, read the first peak of $g(r)$ at the bin holding 10 m, its
    height the eight pairs there over that bin's shell beside the four pairs
    at 20 m over theirs.
    """
    line   = stack([10.0 * arange(3), zeros(3), zeros(3)], -1)
    flight = laid(
        positions  = concat([line + [0.0, 200.0 * row, 0.0] for row in range(4)]),
        velocities = [10.0, 0.0, 0.0]
    )
    centers = flight.centers
    near    = 8 / centers[5] ** 2

    assert_allclose(
        flight.peak,
        ([centers[5]] * 2, [32 * near / (near + 4 / centers[10] ** 2)] * 2),
        rtol = 1e-5
    )


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
    Asserts that from `turn_at_s` on, the first drone commands the turned
    velocity, and the turn's speed and reach read as finite values.
    """
    plan  = short(duration_s=4.0, turn_at_s=1.0, turn_deg=90.0)
    flown = Flight.flown(composite, AIR, plan)

    speed, reach = flown.turn_response

    assert_allclose(
        flown.frames.commands[:, 10:, 0],
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


def test_the_largest_group_reads_its_own_shape_apart_from_a_stray_pair():
    """
    Asserts that drones in pairs on each axis at 30, 20, and 10 m from the
    center read axes of 1 to 2 to 3 as the largest group, beside a pair of
    drones 500 m off that the hop of three launch spacings leaves apart.
    """
    spots = concat(
        [
            diag([30.0, 20.0, 10.0]),
            -diag([30.0, 20.0, 10.0]),
            [[500.0, 0.0, 0.0], [510.0, 0.0, 0.0]]
        ]
    )

    assert_allclose(
        laid(spots, [10.0, 0.0, 0.0]).group_aspect_ratios,
        [[1.0, 2.0, 3.0]] * 2,
        rtol = 1e-5
    )


def test_the_largest_group_s_shape_averages_the_frames_a_shape_interval_apart():
    """
    Asserts that a flock holding axes of 1 to 2 to 3 through the first
    second and 1 to 1 to 3 through the next reads their mean, 1 to 1.5 to 3,
    from its settled frames one second apart.
    """
    first  = concat([diag([30.0, 20.0, 10.0]), -diag([30.0, 20.0, 10.0])])
    second = concat([diag([30.0, 10.0, 10.0]), -diag([30.0, 10.0, 10.0])])
    spots  = where((arange(20) < 10)[:, None, None], first, second)

    assert_allclose(
        laid(spots, [10.0, 0.0, 0.0]).group_aspect_ratios,
        [[1.0, 1.5, 3.0]] * 2,
        rtol = 1e-5
    )


def test_the_radio_s_range_reaches_the_replay_and_the_flight():
    """
    Asserts that two strings of drones 100 m apart replay otherwise under a
    range shorter than the gap than under one spanning it, and that a short
    flight flies otherwise under a range shorter than its spacing.
    """
    string = stack([10.0 * arange(6), zeros(6), zeros(6)], -1)
    flight = laid(concat([string, string + [0.0, 100.0, 0.0]]), [10.0, 0.0, 0.0])
    reach  = {
        range_m: replace(
            STILL,
            cohesion = Centroid(),
            radio    = Radio(delay_s=0.0, loss=0.0, range_m=range_m)
        )
        for range_m in (50.0, 1e4)
    }

    assert not allclose(flight.drifts(reach[50.0], AIR), flight.drifts(reach[1e4], AIR))
    assert not allclose(
        Flight.flown(
            Composite.from_settings(Settings(radio=Radio(range_m=5.0))),
            AIR,
            short()
        ).frames.positions,
        Flight.flown(
            Composite.from_settings(Settings(radio=Radio(range_m=1e4))),
            AIR,
            short()
        ).frames.positions
    )


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
