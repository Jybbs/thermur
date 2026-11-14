"""
Pins each component against its source's form, and the property every one
of them keeps, meaning that moving a drone another drone cannot sense leaves
the second drone's action as it stands.
"""

from common.drones import flock_of, still_air, viewed
from equinox       import filter_jit, tree_at
from hypothesis    import assume, given, strategies
from jax           import Array, lax, tree, vmap
from jax.numpy     import append, array, exp, sqrt
from jax.random    import key, normal
from numpy.testing import assert_allclose
from pytest        import mark, param

from thermur.controller.components import (
    Anisotropic,
    Centered,
    Centroid,
    Consensus,
    Constant,
    Cruise,
    Delayed,
    Elastic,
    Fitted,
    HardSphere,
    Home,
    Intrinsic,
    Metric,
    Overdamped,
    Persistent,
    Vectorial,
    Weighted,
    White,
    Zig
)
from thermur.controller.schemas import Component, Composite, Inbox, Noise, Observation, nearest
from thermur.environment.air    import Air
from thermur.environment.plant  import Plant
from thermur.metrics.schemas    import Plan
from thermur.metrics.trials     import Trial
from thermur.pipeline.settings  import Settings

COMPOSITES = {
    name: composite
    for trial in Trial.listed()
    for name, composite in trial.composites(Composite.from_settings(Settings())).items()
}

AIR = still_air()

NEIGHBORS = Plan().neighbors

DRONES = 2 * NEIGHBORS + 2


@filter_jit
def acted(
    air        : Air,
    composite  : Composite,
    positions  : Array,
    velocities : Array
) -> Array:
    """
    Computes the last drone's command rate followed by the amplitude of its
    command's noise, each drone sensing as many nearest neighbors in `air`
    within the radio's range as a flight does, having heard each one's state
    at once.
    """
    flock  = flock_of(positions, velocities)
    sensed = Observation.sensed(
        flock,
        Inbox.from_broadcast(flock, composite.radio.range_m, flock),
        air,
        0.0,
        NEIGHBORS
    )
    return append(
        vmap(composite)(sensed)[-1],
        vmap(composite.noise.amplitudes)(sensed)[0][-1]
    )


@mark.parametrize("composite", COMPOSITES.values(), ids=COMPOSITES.keys())
@given(
    distance  = strategies.floats(0.0, 500.0),
    direction = strategies.tuples(*[strategies.floats(-1.0, 1.0)] * 3).filter(
        lambda vector: sum(value**2 for value in vector) > 0.01
    ),
    pick = strategies.integers(0, DRONES - 2),
    seed = strategies.integers(0, 2**16)
)
def test_moving_a_drone_the_last_drone_cannot_hear_leaves_an_action_as_it_stands(
    composite : Composite,
    direction : tuple[float, float, float],
    distance  : float,
    pick      : int,
    seed      : int
):
    """
    Pins locality under the default composite and every candidate, moving a
    drone beyond the radio's range of the last drone to a point farther from
    it than both that range and its farthest neighbor, and asserting that
    the last drone's acceleration and the amplitude of its noise hold.
    """
    positions  = 40 * normal(key(seed), (DRONES, 3))
    velocities = 10 * normal(key(seed + 1), (DRONES, 3))
    apart      = sqrt(((positions - positions[-1]) ** 2).sum(-1))
    unheard    = [
        drone for drone in range(DRONES - 1) if apart[drone] > composite.radio.range_m
    ]
    assume(unheard)
    reach = max(
        float(apart[nearest(positions, NEIGHBORS)[-1]].max()),
        composite.radio.range_m
    )
    heading = array(direction) / sqrt((array(direction) ** 2).sum())
    moved   = positions.at[unheard[pick % len(unheard)]].set(
        positions[-1] + (reach + 1.0 + distance) * heading
    )

    assert_allclose(
        acted(
            air        = AIR,
            composite  = composite,
            positions  = moved,
            velocities = velocities
        ),
        acted(
            air        = AIR,
            composite  = composite,
            positions  = positions,
            velocities = velocities
        ),
        atol = 1e-5
    )


@mark.parametrize(
    ("component", "observation", "expected"),
    [
        param(
            Anisotropic(across=1.0, coupling_per_s=0.3),
            viewed([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
            [0.3, 0.6],
            id = "anisotropic"
        ),
        param(
            Constant(coupling_per_s=0.6),
            viewed([[10.0, 0.0, 0.0], [0.0, 30.0, 0.0]]),
            [0.6, 0.6],
            id = "constant"
        ),
        param(
            Metric(coupling_per_s=0.6, onset_k=5.0, range_m=25.0),
            viewed([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]], excess=10.0),
            [0.6, 0.0],
            id = "metric-hot"
        ),
        param(
            Metric(coupling_per_s=0.6, onset_k=5.0, range_m=25.0),
            viewed([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]]),
            [0.6, 0.6],
            id = "metric-cool"
        ),
        param(
            Metric(coupling_per_s=0.6, onset_k=5.0, range_m=25.0),
            viewed([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]], excess=5.0),
            [0.6, 0.0],
            id = "metric-at-onset"
        ),
        param(
            Metric(coupling_per_s=0.6, onset_k=5.0, range_m=25.0),
            viewed([[10.0, 0.0, 0.0], [25.0, 0.0, 0.0]], excess=10.0),
            [0.6, 0.6],
            id = "metric-at-range"
        ),
        param(
            Weighted(coupling_per_s=1.2, range_m=20.0),
            viewed([[20.0, 0.0, 0.0]]),
            [1.2 * exp(-1.0)],
            id = "weighted"
        )
    ]
)
def test_each_coupling_weights_its_neighbors_by_its_source_s_form(
    component   : Component,
    expected    : list[float],
    observation : Observation
):
    """
    Asserts that each coupling sets the rate at which a drone aligns with
    each neighbor as the system it comes from does, by rank, bearing,
    distance, or the heat of the air.
    """
    assert_allclose(component(observation), expected, rtol=1e-6)


@mark.parametrize(
    ("component", "observation", "expected"),
    [
        param(
            Centroid(stiffness_per_s2=0.1),
            viewed([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
            [0.5, 0.5, 0.0],
            id = "centroid"
        ),
        param(
            Consensus(centering=0.5),
            viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]])),
            [0.0, 1.5, 0.0],
            id = "consensus"
        ),
        param(
            Cruise(cruise_m_s=10.0, relaxation_s=2.0),
            viewed([[10.0, 0.0, 0.0]], (0.0, 6.0, 0.0)),
            [0.0, 2.0, 0.0],
            id = "cruise"
        ),
        param(
            Delayed(),
            viewed([[10.0, 0.0, 0.0]], delayed=array([[10.0, -1.0, 0.0]])),
            [0.0, -1.0, 0.0],
            id = "delayed"
        ),
        param(
            Home(height_m=100.0, stiffness_per_s2=0.5),
            viewed([[10.0, 0.0, 0.0]], position=array([1000.0, 1000.0, 130.0])),
            [0.0, 0.0, -15.0],
            id = "home-heading-for-the-rim"
        ),
        param(
            Elastic(rest_m=18.0, stiffness_per_s2=0.024),
            viewed([[20.0, 0.0, 0.0]]),
            [0.048, 0.0, 0.0],
            id = "elastic"
        ),
        param(
            Fitted(
                bearing          = 0.5,
                range_m          = 40.0,
                rest_m           = 18.0,
                stiffness_per_s2 = 0.024
            ),
            viewed([[40.0, 0.0, 0.0]]),
            [0.024 * 22.0 / 2.0 * 1.5, 0.0, 0.0],
            id = "fitted"
        ),
        param(
            HardSphere(sphere_m=1.0, spread_m=2.0, weight_m_s2=8.0),
            viewed([[3.0, 0.0, 0.0], [0.0, 0.5, 0.0]]),
            [-4.0 * exp(-0.5), -4.0, 0.0],
            id = "hard-sphere"
        ),
        param(
            HardSphere(sphere_m=1.0, spread_m=2.0, weight_m_s2=8.0),
            viewed([[3.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]),
            [-4.0 * exp(-0.5), -4.0, 0.0],
            id = "hard-sphere-past-an-unfilled-rank"
        ),
        param(
            Overdamped(),
            viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]])),
            [0.0, 1.0, 0.0],
            id = "overdamped"
        ),
        param(
            Centered(blind_deg=90.0, sphere_m=0.2, weight_m_s2=12.0),
            viewed([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
            [2.0, 2.0, 0.0],
            id = "centered-at-the-border"
        ),
        param(
            Centered(blind_deg=90.0, sphere_m=0.2, weight_m_s2=12.0),
            viewed(
                [
                    [10.0, 0.0, 0.0],
                    [-10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]
            ),
            [2.0, 2.0, 0.0],
            id = "centered-past-an-unfilled-rank"
        ),
        param(
            Centered(blind_deg=90.0, sphere_m=0.2, weight_m_s2=12.0),
            viewed(
                [
                    [10.0, 0.0, 0.0],
                    [-10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, -10.0, 0.0]
                ]
            ),
            [0.0, 0.0, 0.0],
            id = "centered-inside"
        ),
        param(
            Centered(blind_deg=90.0, sphere_m=5.0, weight_m_s2=12.0),
            viewed([[10.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            [12.0 * 2**0.5 / 4, 0.0, 0.0],
            id = "centered-past-the-hard-sphere"
        ),
        param(
            Centered(blind_deg=90.0, sphere_m=0.2, weight_m_s2=12.0),
            viewed(
                [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                centrality = array(0.5)
            ),
            [3.0, 3.0, 0.0],
            id = "centered-on-every-held-sender"
        )
    ]
)
def test_each_force_follows_its_source_s_form(
    component   : Component,
    expected    : list[float],
    observation : Observation
):
    """
    Asserts that each heading, cohesion, and contact accelerates a drone as
    the system it comes from does, from the neighbors' offsets, velocities,
    and delayed velocities.
    """
    assert_allclose(component(observation), expected, atol=1e-6, rtol=1e-5)


@mark.parametrize(
    ("noise", "observation", "expected"),
    [
        param(
            Intrinsic(copying_per_s=1.0, spontaneous=3.7),
            viewed([[10.0, 0.0, 0.0]]),
            (3.7, 0.0),
            id = "intrinsic-agreeing"
        ),
        param(
            Intrinsic(copying_per_s=1.0, spontaneous=3.7),
            viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 3.0, 0.0]])),
            (sqrt(3.7**2 + 3.0), 0.0),
            id = "intrinsic-disagreeing"
        ),
        param(
            Persistent(persistence_s=0.5, strength_m_s2=3.0),
            viewed([[10.0, 0.0, 0.0]]),
            (0.0, 6.0),
            id = "persistent"
        ),
        param(
            Vectorial(amplitude=4.1, floor=0.3),
            viewed([[10.0, 0.0, 0.0]]),
            (4.1, 0.0),
            id = "vectorial-ordered"
        ),
        param(
            Vectorial(amplitude=4.1, floor=0.3),
            viewed(
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                neighbors = array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
            ),
            (4.1 / 0.3, 0.0),
            id = "vectorial-disordered"
        ),
        param(White(amplitude=4.2), viewed([[10.0, 0.0, 0.0]]), (4.2, 0.0), id="white"),
        param(Zig(amplitude=4.2), viewed([[10.0, 0.0, 0.0]]), (4.2, 0.0), id="zig")
    ]
)
def test_each_noise_scales_its_increments_by_its_source_s_form(
    expected    : tuple[float, float],
    noise       : Noise,
    observation : Observation
):
    """
    Asserts that each noise scales the Brownian increment on a drone's
    velocity and on its force as the system it comes from does.
    """
    assert_allclose(noise.amplitudes(observation), expected, rtol=1e-6)


def test_a_copied_zig_starts_at_once_where_the_radio_s_delay_outlasts_the_cue():
    """
    Asserts that a drone copying a zig the radio delivers later than its
    reaction time and the cue's starts the zig at once, its wait held at
    zero.
    """
    assert_allclose(
        ZIG.excited(
            delay_s     = 3.0,
            draw        = array(0.0),
            observation = viewed(
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                forces = array([[4.5, -1.0, 0.0], [0.0, 0.0, 0.0]])
            ),
            step_s = 0.1,
            turn   = array([0.0, 0.0, 0.0])
        ),
        (5.0, -1.0, 0.0),
        atol = 1e-6
    )


def test_a_persistent_force_enters_the_acceleration_and_decays():
    """
    Asserts that the persistent noise adds the force a drone holds to its
    acceleration and relaxes that force over its persistence time, where
    every other noise adds and relaxes nothing.
    """
    force = array([1.0, -2.0, 0.5])
    view  = viewed([[10.0, 0.0, 0.0]], force=force)

    assert_allclose(Persistent(persistence_s=2.0)(view), force)
    assert_allclose(Persistent(persistence_s=2.0).relaxation(view), -force / 2.0)
    assert_allclose(White()(view), 0.0)
    assert_allclose(White().relaxation(view), 0.0)


ZIG = Zig(
    accel_m_s2   = 2.0,
    back_s       = 1.0,
    branching    = 1.0,
    cue_s        = 0.5,
    reaction_s   = 1.5,
    refractory_s = 3.0,
    side_s       = 1.0
)
"""
The zig whose push lasts from a countdown of 5 s to one of 3 s, its side
half from 5 s to 4 s.
"""


@mark.parametrize(
    ("force", "expected"),
    [
        param((4.5, 1.0, 0.0), (0.0, 2.0, 0.0), id="side"),
        param((4.5, -1.0, 0.0), (0.0, -2.0, 0.0), id="side-to-the-right"),
        param((3.5, 1.0, 0.0), (0.0, -2.0, 0.0), id="back"),
        param((6.0, 1.0, 0.0), (0.0, 0.0, 0.0), id="waiting"),
        param((2.0, 1.0, 0.0), (0.0, 0.0, 0.0), id="refractory"),
        param((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), id="resting")
    ]
)
def test_a_zig_pushes_to_its_side_and_back_across_the_heading(
    expected : tuple[float, float, float],
    force    : tuple[float, float, float]
):
    """
    Asserts that a zig pushes a drone flying east toward its side for
    `side_s` and away from it for `back_s`, and adds nothing through the
    wait before the zig and the refractory time after it.
    """
    assert_allclose(ZIG(viewed([[10.0, 0.0, 0.0]], force=array(force))), expected)


@mark.parametrize(
    ("force", "forces", "turn", "draw", "expected"),
    [
        param(
            (0.0, 0.0, 0.0),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            (0.0, -2.0, 0.0),
            0.0,
            (6.5, -1.0, 0.0),
            id = "walled"
        ),
        param(
            (0.0, 0.0, 0.0),
            [[4.5, -1.0, 0.0], [0.0, 0.0, 0.0]],
            (0.0, 0.0, 0.0),
            0.0,
            (6.0, -1.0, 0.0),
            id = "copied"
        ),
        param(
            (0.0, 0.0, 0.0),
            [[4.5, -1.0, 0.0], [4.5, 1.0, 0.0]],
            (0.0, 0.0, 0.0),
            0.0,
            (6.0, 1.0, 0.0),
            id = "tied"
        ),
        param(
            (0.0, 0.0, 0.0),
            [[4.5, -1.0, 0.0], [0.0, 0.0, 0.0]],
            (0.0, 0.0, 0.0),
            0.999,
            (0.0, 0.0, 0.0),
            id = "uncaught"
        ),
        param(
            (0.0, 0.0, 0.0),
            [[4.5, -1.0, 0.0], [0.0, 0.0, 0.0]],
            (0.0, 1.9, 1.0),
            0.999,
            (0.0, 0.0, 0.0),
            id = "turned-short-of-the-zig"
        ),
        param(
            (2.0, 1.0, 0.0),
            [[4.5, -1.0, 0.0], [0.0, 0.0, 0.0]],
            (0.0, -2.0, 0.0),
            0.0,
            (1.9, 1.0, 0.0),
            id = "busy"
        ),
        param(
            (0.05, 1.0, 0.0),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            (0.0, 0.0, 0.0),
            0.0,
            (0.0, 1.0, 0.0),
            id = "settling"
        )
    ]
)
def test_a_zig_starts_at_the_rim_or_from_a_zigging_neighbor(
    draw     : float,
    expected : tuple[float, float, float],
    force    : tuple[float, float, float],
    forces   : list[list[float]],
    turn     : tuple[float, float, float]
):
    """
    Asserts that a resting drone starts its zig away from the rim once the
    roost turns it as hard as the zig would, after its reaction time, or
    copies a zigging neighbor's side after its reaction time and the cue's
    less the radio's delay where the draw falls below the chance, and that a
    drone already counting down counts down by the step.
    """
    assert_allclose(
        ZIG.excited(
            delay_s     = 1.0,
            draw        = array(draw),
            observation = viewed(
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                force  = array(force),
                forces = array(forces)
            ),
            step_s = 0.1,
            turn   = array(turn)
        ),
        expected,
        atol = 1e-6
    )


def test_a_zig_s_chance_of_spreading_grows_with_each_zigging_neighbor():
    """
    Asserts that a resting drone hearing two of its seven neighbors zig
    through one whole zig copies them where the draw of 0.2 falls below the
    chance of 1 - (6/7)**2, about 0.265, which one zigging neighbor's 1/7
    would leave it short of.
    """
    assert_allclose(
        ZIG.excited(
            delay_s     = 1.0,
            draw        = array(0.2),
            observation = viewed(
                [[10.0 * (index + 1), 0.0, 0.0] for index in range(7)],
                forces = array([[4.5, -1.0, 0.0]] * 2 + [[0.0, 0.0, 0.0]] * 5)
            ),
            step_s = ZIG.zig_s,
            turn   = array([0.0, 0.0, 0.0])
        ),
        (6.0, -1.0, 0.0),
        atol = 1e-6
    )


@mark.parametrize(("margin", "copied"), [(-1e-4, True), (1e-4, False)])
def test_one_zigging_neighbor_sets_a_drone_off_at_the_branching_ratio_over_the_count(
    copied : bool,
    margin : float
):
    """
    Asserts that one zigging neighbor among seven sets a resting drone off
    within one whole zig with the chance `branching` over seven.
    """
    result = ZIG.excited(
        delay_s     = 0.0,
        draw        = array(1.0 / 7 + margin),
        observation = viewed(
            [[10.0, float(rank), 0.0] for rank in range(7)],
            forces = array([[4.5, 1.0, 0.0]] + [[0.0, 0.0, 0.0]] * 6)
        ),
        step_s = ZIG.side_s + ZIG.back_s,
        turn   = array([0.0, 0.0, 0.0])
    )

    assert bool(result[0] > 0) is copied


@mark.parametrize(
    ("offsets", "forces"),
    [
        param(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[4.5, 1.0, 0.0], [0.0, 0.0, 0.0]],
            id = "unfilled-rank"
        ),
        param(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[6.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            id = "waiting-neighbor"
        ),
        param(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            id = "refractory-neighbor"
        )
    ]
)
def test_a_zig_copies_no_neighbor_short_of_its_push(
    forces  : list[list[float]],
    offsets : list[list[float]]
):
    """
    Asserts that a rank no sender fills, a neighbor waiting to zig, and a
    neighbor in its refractory time set no drone off.
    """
    assert_allclose(
        ZIG.excited(
            delay_s     = 0.0,
            draw        = array(0.0),
            observation = viewed(
                offsets,
                force  = array([0.0, 0.0, 0.0]),
                forces = array(forces)
            ),
            step_s = 0.1,
            turn   = array([0.0, 0.0, 0.0])
        ),
        0.0
    )


@mark.parametrize(
    ("position", "velocity", "expected"),
    [
        param(
            (1000.0, 1500.0, 100.0),
            (10.0, 0.0, 0.0),
            (0.0, -0.94 * 10.0**2 / 750_000**0.5, 0.0),
            id = "inside"
        ),
        param(
            (2500.0, 1000.0, 100.0),
            (0.0, 10.0, 0.0),
            (-5.0, 0.0, 0.0),
            id = "outside"
        ),
        param((1000.0, 1999.9, 100.0), (10.0, 0.0, 0.0), (0.0, -5.0, 0.0), id="bounded")
    ]
)
def test_home_turns_a_drone_from_the_rim_by_its_time_to_reach_it(
    expected : tuple[float, float, float],
    position : tuple[float, float, float],
    velocity : tuple[float, float, float]
):
    """
    Asserts that the roost turns a drone across its heading away from the
    rim it would reach, by the turn times its speed over the time to reach
    the rim, at most the bound, and by the bound outside the disk.
    """
    assert_allclose(
        Home()(viewed([[1.0, 0.0, 0.0]], velocity, position=array(position))),
        expected,
        atol = 1e-4,
        rtol = 1e-4
    )


@mark.parametrize(
    ("outward", "ground", "expected"),
    [
        param((0.0, 0.0), (10.0, 0.0), 100.0, id="center"),
        param((500.0, 0.0), (10.0, 0.0), 50.0, id="ahead"),
        param((500.0, 0.0), (-10.0, 0.0), 150.0, id="behind"),
        param((0.0, 0.0), (0.0, 0.0), 0.0, id="still")
    ]
)
def test_the_time_to_reach_the_rim_follows_the_course_to_the_circle(
    expected : float,
    ground   : tuple[float, float],
    outward  : tuple[float, float]
):
    """
    Asserts that the time to reach the rim is the distance along the ground
    course to the circle over the ground speed, and zero for a drone holding
    still.
    """
    assert_allclose(Home().rim_s(array(ground), array(outward)), expected, rtol=1e-6)


@mark.parametrize(
    ("height", "climb", "updraft", "expected"),
    [
        param(100.0, 2.0, 0.0, -2.0, id="climbing"),
        param(104.0, 0.0, 0.0, -1.0, id="above"),
        param(100.0, 0.0, 1.5, -1.5, id="lifted"),
        param(96.0, -1.0, 0.0, 2.0, id="below-and-sinking")
    ]
)
def test_home_damps_a_drone_s_height_critically(
    climb    : float,
    expected : float,
    height   : float,
    updraft  : float
):
    """
    Asserts that the roost pulls a drone toward its height by the stiffness
    times the height away and against its climb over the ground, the wind's
    included, by twice the stiffness's square root.
    """
    assert_allclose(
        Home(height_m=100.0, stiffness_per_s2=0.25)(
            viewed(
                [[1.0, 0.0, 0.0]],
                (10.0, 0.0, climb),
                air = {"excess": array(0.0), "u": array(0.0), "v": array(0.0)}
                | {"w": array(updraft)},
                position = array([1000.0, 1000.0, height])
            )
        )[2],
        expected,
        atol = 1e-6
    )


def test_home_returns_a_lifted_drone_to_its_height_without_passing_it():
    """
    Asserts that a drone the roost alone moves, lifted 30 m above the
    roost's height at the drone scale, settles back to within a meter of
    it in ten minutes without ever falling below it, the pull setting the
    rate of the command and the velocity following the command through the
    plant's lag, as in flight, where a roost left undamped swings the drone
    more than 50 m below its height.
    """
    plant = Plant()
    roost = Settings().roost.scaled(Plan().similarity)
    alone = viewed([[1.0, 0.0, 0.0]])

    def step(state: Array, *_: None) -> tuple[Array, Array]:
        """
        Advances the height, the commanded climb, and the climb by a tenth
        of a second under the roost's vertical pull.
        """
        height, command, climb = state
        pull = roost.vertical(
            tree_at(
                lambda seen: (seen.position, seen.velocity),
                alone,
                (array([1000.0, 1000.0, height]), array([10.0, 0.0, climb]))
            )
        )
        rate = plant.tracked(array([10.0, 0.0, command]), array([10.0, 0.0, climb]))
        return (
            array([height + climb * 0.1, command + pull * 0.1, climb + rate[2] * 0.1]),
            height
        )

    _, heights = lax.scan(
        step, array([roost.height_m + 30.0, 0.0, 0.0]), length=6000
    )

    assert heights.min() >= roost.height_m - 1e-3
    assert_allclose(heights[-1], roost.height_m, atol=1.0)


@mark.parametrize(
    ("component", "expected"),
    [
        param(
            Constant(coupling_per_s=2.0),
            Constant(coupling_per_s=0.5),
            id = "constant"
        ),
        param(
            HardSphere(sphere_m=0.5, spread_m=1.0, weight_m_s2=8.0),
            HardSphere(sphere_m=2.0, spread_m=4.0, weight_m_s2=2.0),
            id = "hard-sphere"
        ),
        param(
            Centered(blind_deg=90.0, sphere_m=0.5, weight_m_s2=8.0),
            Centered(blind_deg=90.0, sphere_m=2.0, weight_m_s2=2.0),
            id = "centered"
        ),
        param(Consensus(), Consensus(), id="consensus"),
        param(Cruise(relaxation_s=1.0), Cruise(relaxation_s=4.0), id="cruise"),
        param(Home(stiffness_per_s2=8.0), Home(stiffness_per_s2=0.5), id="home"),
        param(White(amplitude=4.0), White(amplitude=2.0), id="white"),
        param(
            Zig(
                accel_m_s2   = 8.0,
                amplitude    = 4.0,
                back_s       = 0.5,
                cue_s        = 0.25,
                reaction_s   = 0.5,
                refractory_s = 1.0,
                side_s       = 0.25
            ),
            Zig(
                accel_m_s2   = 2.0,
                amplitude    = 2.0,
                back_s       = 2.0,
                cue_s        = 1.0,
                reaction_s   = 2.0,
                refractory_s = 4.0,
                side_s       = 1.0
            ),
            id = "zig"
        ),
        param(Overdamped(), Overdamped(), id="overdamped"),
        param(Centroid(), Centroid(), id="centroid")
    ]
)
def test_each_slot_carried_to_a_flock_four_times_the_size_scales_its_values(
    component : Component,
    expected  : Component
):
    """
    Asserts that each slot holding a starling flock's values multiplies its
    lengths by the factor, divides its accelerations and rates by it, its
    stiffness by its square, and its noise by its square root, leaving every
    slot holding a drone flock's own values.
    """
    assert_allclose(tree.leaves(component.scaled(4.0)), tree.leaves(expected))
