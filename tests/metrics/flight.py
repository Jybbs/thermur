"""
Pins `Flight`, meaning that a flight repeats from its seed, that its batch
flies each seed as one flight alone would, the tables a recorded run saves,
the first drone's turn, one step moving a flock along its old velocities,
the replay ranking composites, and the helpers each statistic reads.
"""

from common.drones import flock_of, short, still_air
from dataclasses   import replace
from jax           import Array, jit
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
    sort,
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
    InertialSpin,
    Marginal,
    Overdamped,
    Persistent,
    White
)
from thermur.controller.schemas import Composite, Flock, Inbox, Radio, across
from thermur.environment.plant  import Plant
from thermur.environment.wrf    import WrfOutput
from thermur.metrics.flight     import Flight, first_zero, kept, ratio_on_hull, response, stepped
from thermur.metrics.schemas    import Plan
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
    speed    = Marginal(fixed=True)
)
"""
The composite whose command rate is zero on every flock, every drone hearing
each other at once.
"""

PUSHED = Composite(
    cohesion = Centroid(stiffness_per_s2=0.0),
    contact  = HardSphere(sphere_m=5.0, weight_m_s2=250.0),
    coupling = Constant(coupling_per_s=0.0),
    heading  = InertialSpin(friction=1.0, inertia_s=0.02),
    noise    = White(amplitude=0.0),
    plant    = Plant(),
    radio    = Radio(delay_s=0.0, loss=0.0),
    roost    = Centroid(stiffness_per_s2=0.0),
    speed    = Marginal(amplitude=0.0)
)
"""
The composite under which two drones 3 m apart, inside each other's hard
sphere, push each other apart at 250 m/s^2, and the friction stops a spin
within one step of 0.02 s, with no noise, no alignment, and no roost.
"""

EAST = array([[10.0, 0.0, 0.0]] * 9)
"""
The velocities of nine drones flying east at 10 m/s.
"""

SPOTS = array([[1000.0, 1000.0 + 200.0 * row, 100.0] for row in range(9)])
"""
A line of nine drones 200 m apart across the course, inside the still air.
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
def hundred(composite: Composite, request: FixtureRequest) -> Flight:
    """
    Flies the default plan at 100 drones under the default composite through
    the parametric threat's air, for the plan's two minutes and for the five
    the rules page reads.
    """
    return threatened(composite, 100, request.param)


@fixture(scope="module")
def flight(composite: Composite) -> Flight:
    """
    Flies the short plan under the default composite through still air.
    """
    return Flight.flown(AIR, composite, short())


class Drawn(White):
    """
    White noise holding as its force the uniform draw each step reads.
    """

    def excited(self, draw: Array, **_: object) -> Array:
        """
        Holds `draw` east as the force.
        """
        return array([1.0, 0.0, 0.0]) * draw


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


def threatened(composite: Composite, drones: int, duration_s: float) -> Flight:
    """
    Flies the default plan at `drones` for `duration_s` under `composite`
    through the parametric threat's air.
    """
    settings = Settings()
    return Flight.flown(
        WrfOutput.from_settings(settings).air,
        composite,
        settings.plan.changed(drones=drones, duration_s=duration_s)
    )


def pushed(flock: Flock) -> Array:
    """
    Steps a flock laid out by hand once under `PUSHED` through still air,
    from where it stands and under no Brownian increment, reading each
    drone's command at the step's end.
    """
    still = zeros(flock.positions.shape)
    return stepped(
        air       = AIR,
        composite = PUSHED,
        flock     = flock,
        inbox     = Inbox.from_broadcast(flock, PUSHED.radio.range_m, flock),
        key       = key(0),
        path      = lambda *_: flock_of(still, still),
        plan      = Plan(drones=len(flock.positions)),
        t         = array(0.0)
    ).commands


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
            AIR,
            replace(starling, heading=Delayed(lag_s=steps * plan.step_s)),
            plan
        ).frames.velocities,
        Flight.flown(AIR, replace(starling, heading=shorter), plan).frames.velocities,
        atol = 1e-3
    )


@mark.slow
def test_a_flight_of_100_drones_holds_the_roost_s_height(hundred: Flight):
    """
    Asserts that a flight of 100 drones holds them within 10 m of the
    roost's height on the mean over its settled frames and its seeds, where
    a plant bounding a descent tighter than a climb lifts the flock some 60
    m above that height in two minutes.
    """
    assert_allclose(
        asarray(hundred.settled.positions)[..., 2].mean(),
        Settings().roost.height_m,
        atol = 10.0
    )


@mark.slow
def test_a_flight_of_100_drones_keeps_a_starling_flock_s_shape(hundred: Flight):
    """
    Asserts that the largest group of a flight of 100 drones, averaged over
    the frames `Plan.shape_every_s` apart through two minutes and through
    five and over the seeds, holds its longest axis within 4.02 to 6.94
    times its thinnest, the span of the starling flocks the starling model
    is checked against.

    The middle axis goes unchecked, since that ratio rises with flock size
    in the starling model's flocks, a dependence the Rome flocks behind the
    span do not show, whereas the longest ratio depends on size in neither.
    The mean over time and seeds is what the span bounds, since the model's
    flocks report their shape as a mean over time, and one frame's shape
    stretches and folds as the flock turns at the roost's rim.
    """
    assert 4.02 <= hundred.group_aspect_ratios.mean(0)[2] <= 6.94


@mark.slow
@mark.parametrize(
    ("drones", "duration_s"),
    [
        param(448, 120.0, id="448-for-two-minutes"),
        param(448, 300.0, id="448-for-five-minutes"),
        param(1000, 120.0, id="1000-for-two-minutes")
    ]
)
def test_a_larger_flight_keeps_a_starling_flock_s_shape(
    composite  : Composite,
    drones     : int,
    duration_s : float
):
    """
    Asserts that a flight of 448 drones, the smallest of the starling flocks
    the span is read from, and one of 1000 hold the largest group's longest
    axis within 4.02 to 6.94 times its thinnest, averaged as at 100 drones.
    """
    flight = threatened(composite, drones, duration_s)

    assert 4.02 <= flight.group_aspect_ratios.mean(0)[2] <= 6.94


@mark.slow
def test_a_flight_of_100_drones_sets_off_zigs(hundred: Flight):
    """
    Asserts that some drone of a flight of 100 drones after two minutes and
    after five holds a zig's countdown at some kept frame, so the roost's
    turn and the neighbors' zigs reach the noise slot's state in flight.
    """
    assert (asarray(hundred.frames.forces)[..., 0] > 0).any()


@mark.slow
def test_a_flight_of_100_drones_keeps_the_outdoor_drone_flock_s_spacing(
    hundred: Flight
):
    """
    Asserts that a flight of 100 drones after two minutes and after five
    keeps its drones 12 to 30 m from their nearest neighbors on the mean,
    the spacing the outdoor drone flock kept.
    """
    assert 12.0 <= float(hundred.spacing_m.mean()) <= 30.0


@mark.slow
def test_a_flight_of_100_drones_keeps_every_drone_over_the_roost(hundred: Flight):
    """
    Asserts that every drone of a flight of 100 drones through two minutes
    and through five sits within the roost's radius of its center and above
    the ground at every kept frame.
    """
    roost = Settings().roost
    spots = asarray(hundred.frames.positions)

    assert (
        linalg.norm(spots[..., :2] - asarray(roost.center_m), axis=-1) <= roost.radius_m
    ).all()
    assert (spots[..., 2] >= 0).all()


def test_a_flight_holds_every_command_within_the_plant_s_speed_bounds():
    """
    Asserts that noise strong enough to carry a command past the plant's
    speed bounds within a step leaves every command at every kept frame
    within them, some at a bound, so the command stops at the speed the
    plant can fly rather than integrating past it.
    """
    plant    = Plant()
    commands = Flight.flown(
        AIR,
        replace(STILL, heading=Consensus(), noise=White(amplitude=30.0)),
        short(turn_deg=0.0)
    ).frames.commands
    assert (linalg.norm(commands[..., :2], axis=-1) <= plant.level_m_s + 1e-4).all()
    assert (commands[..., 2] <= plant.climb_m_s + 1e-4).all()
    assert (commands[..., 2] >= -plant.descent_m_s - 1e-4).all()
    assert allclose(abs(commands[..., 2]).max(), plant.climb_m_s)


def test_a_flight_repeats_from_its_seed_and_differs_across_seeds(
    composite : Composite,
    flight    : Flight
):
    """
    Asserts that flying one plan again keeps every frame, that each seed
    of the batch flies a flock of its own, and that another `seed` flies
    another batch.
    """
    again   = Flight.flown(AIR, composite, short())
    another = Flight.flown(AIR, composite, short(seed=1))

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
        flight.drifts(AIR, replace(STILL, coupling=Constant())),
        0.0,
        atol = 1e-5
    )
    assert_allclose(
        flight.drifts(AIR, delayed),
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


def test_a_persistent_force_leaves_zero_in_flight(composite: Composite):
    """
    Asserts that a flight under the persistent force holds a nonzero force
    on some drone at some kept frame, so each step advances the force the
    step before it left rather than the one it started from.
    """
    assert (
        asarray(
            Flight.flown(
                AIR,
                replace(composite, noise=Persistent()),
                short()
            ).frames.forces
        ) != 0
    ).any()


def test_a_push_across_the_command_keeps_each_drone_s_speed():
    """
    Asserts that two drones a contact pushes apart across their command keep
    the 10 m/s they commanded, the speed being the stepped command's part
    along the heading rather than its length of `sqrt(10^2 + 5^2)`.
    """
    commands = pushed(flock_of(SPOTS.at[1, 1].set(SPOTS[0, 1] + 3.0), EAST))

    assert_allclose(linalg.norm(commands[:2], axis=-1), 10.0, rtol=1e-5)
    assert abs(commands[1, 1]) > 1.0


def test_a_relaxing_mode_over_a_static_deformation_peaks_in_the_lowest_bin():
    """
    Asserts that a line of drones whose velocities across the course hold
    a static deformation five times the size of a fluctuation relaxing
    with no wave reads its spectral peak at the lowest frequency the flight
    resolves, since the deformation is the mode's time mean and leaves the
    zero bin once that mean is removed.
    """
    spots  = stack([10.0 * arange(12), zeros(12), zeros(12)], -1)
    along  = spots[:, 0] - spots[:, 0].mean()
    wave   = 2 * pi * along / (4 * along.std())
    swing  = 5 * cos(wave) + exp(-arange(20)[:, None] / 3) * sin(wave)
    flight = laid(
        positions  = spots,
        velocities = stack([full(swing.shape, 10.0), swing, zeros(swing.shape)], -1)
    )

    assert_allclose(
        flight.spectral_peak_hz,
        1 / (flight.settled.velocities.shape[1] * flight.plan.frame_s),
        rtol = 1e-5
    )


def test_a_replay_ranks_every_composite_it_is_given(
    composite : Composite,
    flight    : Flight
):
    """
    Asserts that replaying the frames gives one drift a frame but the last,
    and one log-likelihood and one coefficient for each composite named.
    """
    ranked = flight.likelihoods(AIR, {"one": composite, "two": composite})

    assert flight.drifts(AIR, composite).shape == (2, 19, 12, 3)
    assert list(ranked.model.values) == ["one", "two"]
    assert isfinite(ranked.log_likelihood).all()
    assert_allclose(ranked.log_likelihood[0], ranked.log_likelihood[1])


@mark.parametrize(
    ("fixed", "held"),
    [param(True, True, id="fixed"), param(False, False, id="free")]
)
def test_a_fixed_speed_keeps_every_drone_at_the_cruise(
    composite : Composite,
    fixed     : bool,
    held      : bool
):
    """
    Asserts that every drone whose climb or descent stays inside the plant's
    bounds commands `cruise_m_s` at every kept frame under a speed slot
    holding the speed fixed, and that the commanded speeds spread under one
    leaving it free, inside the range the confinement holds.
    """
    plan     = short()
    commands = Flight.flown(
        AIR,
        replace(composite, speed=Marginal(fixed=fixed)),
        plan
    ).frames.commands
    speeds = linalg.norm(commands, axis=-1)
    inside = (-composite.plant.descent_m_s < commands[..., 2]) & (
        commands[..., 2] < composite.plant.climb_m_s
    )

    assert allclose(speeds[inside], plan.cruise_m_s, rtol=1e-4) == held
    assert plan.airspeeds_m_s[0] < speeds.min() <= speeds.max() < plan.airspeeds_m_s[1]


@mark.parametrize(
    ("added", "reach"),
    [param(0.0, 0.0, id="rotating"), param(pi / 2, 1.0, id="turned")]
)
def test_a_flock_rotating_as_a_whole_reads_only_the_rotation_its_turn_adds(
    added : float,
    reach : float
):
    """
    Asserts that drones all rotating together read, against an unturned
    flight rotating with them, only the rotation added from the turn on, no
    reach where none is added and the whole turn for a further quarter turn,
    and no speed where every shell joins at once.
    """
    spots            = stack([10.0 * arange(12), zeros(12), zeros(12)], -1)
    whole            = 0.05 * arange(1, 21)[:, None] * ones(12)
    unturned, turned = (
        laid(
            positions  = spots,
            velocities = 10.0 * stack(
                [cos(angles), sin(angles), zeros(angles.shape)],
                -1
            ),
            turn_deg = 90.0
        )
        for angles in (whole, whole + where(arange(20) >= 10, added, 0.0)[:, None])
    )

    assert_allclose(
        replace(turned, unturned=unturned.frames).turn_response,
        [[0.0, 0.0], [reach, reach]],
        atol = 1e-6
    )


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
        flight.likelihoods(AIR, {"still": STILL}).log_likelihood,
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


def test_a_spin_the_friction_stops_within_the_step_leaves_the_heading():
    """
    Asserts that each command turns at the spin the step reaches, so a
    spin of 1 rad/s the friction stops within one step leaves every drone
    commanding east, where turning at the spin the step starts from would
    swing it 0.2 m/s north.
    """
    commands = pushed(
        flock = replace(flock_of(SPOTS, EAST), spins=array([[0.0, 0.0, 1.0]] * 9))
    )

    assert_allclose(commands, EAST, atol=1e-4)


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


def test_each_step_of_a_flight_reads_a_draw_of_its_own(composite: Composite):
    """
    Asserts that a noise slot holding each step's draw as its force holds
    another force at every step, each kept as a frame, so no two steps read
    one draw.
    """
    forces = asarray(
        Flight.flown(
            AIR,
            replace(composite, noise=Drawn()),
            short(frame_every=1)
        ).frames.forces
    )

    assert (diff(sort(forces[..., 0], axis=1), axis=1) != 0).all()


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


def test_one_step_moves_each_drone_along_its_old_velocity_inside_the_plant(
    composite: Composite
):
    """
    Asserts that one step under no Brownian increment moves each drone by
    `step_s` times the velocity it held, and leaves each command inside the
    plant's bounds.
    """
    plan   = short()
    launch = plan.launched(key(plan.seed))
    still  = zeros(launch.velocities.shape)

    moved = stepped(
        air       = AIR,
        composite = composite,
        flock     = launch,
        inbox     = Inbox.from_broadcast(launch, composite.radio.range_m, launch),
        key       = key(plan.seed),
        path      = lambda *_: flock_of(still, still),
        plan      = plan,
        t         = array(0.0)
    )

    assert_allclose(
        moved.positions,
        launch.positions + plan.step_s * launch.velocities,
        rtol = 1e-5
    )
    assert_allclose(composite.plant.held(moved.commands), moved.commands)


def test_the_first_drone_holds_its_turn_and_the_turn_reaches_the_flock(
    composite: Composite
):
    """
    Asserts that from `turn_at_s` on, the first drone commands the turned
    velocity, that the unturned flight in the same batch matches every frame
    before the turn and never turns, and that the turn's speed and reach
    read as finite values.
    """
    plan  = short(duration_s=4.0, turn_at_s=1.0, turn_deg=90.0)
    flown = Flight.flown(AIR, composite, plan)

    speed, reach = flown.turn_response

    assert_allclose(
        flown.frames.commands[:, 10:, 0],
        full((2, 30, 3), plan.turned),
        atol = 1e-5
    )
    assert_array_equal(flown.unturned.positions[:, :10], flown.frames.positions[:, :10])
    assert (
        linalg.norm(flown.unturned.commands[:, 10:, 0] - plan.turned, axis=-1) > 1
    ).all()
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

    assert not allclose(flight.drifts(AIR, reach[50.0]), flight.drifts(AIR, reach[1e4]))
    assert not allclose(
        Flight.flown(
            AIR,
            Composite.from_settings(Settings(radio=Radio(range_m=5.0))),
            short()
        ).frames.positions,
        Flight.flown(
            AIR,
            Composite.from_settings(Settings(radio=Radio(range_m=1e4))),
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
    assert flight.unturned is None
    assert (table.fragments >= 1).all()
