"""
Pins each component against its source's form, and the property every one
of them keeps, meaning that moving a drone another drone cannot sense leaves
the second drone's action as it stands.
"""

from common.drones import flock_of, still_air, viewed
from equinox       import filter_jit
from hypothesis    import given, strategies
from jax           import Array, tree, vmap
from jax.numpy     import append, array, exp, sqrt
from jax.random    import key, normal
from numpy         import diag
from numpy.testing import assert_allclose
from pytest        import mark, param
from scipy.linalg  import solve_continuous_lyapunov

from thermur.controller.components import (
    Anisotropic,
    Centroid,
    Consensus,
    Constant,
    Delayed,
    Elastic,
    Fitted,
    HardSphere,
    Intrinsic,
    Metric,
    Overdamped,
    Peripheral,
    Persistent,
    Roost,
    Vectorial,
    Weighted,
    White
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
    pick = strategies.integers(0, DRONES - NEIGHBORS - 2),
    seed = strategies.integers(0, 2**16)
)
def test_moving_a_drone_outside_the_neighbors_leaves_an_action_as_it_stands(
    composite : Composite,
    direction : tuple[float, float, float],
    distance  : float,
    pick      : int,
    seed      : int
):
    """
    Pins locality under the default composite and every candidate, moving
    a drone outside the last drone's neighbors to a point farther from
    the last drone than its farthest neighbor, and asserting that the last
    drone's acceleration and the amplitude of its noise hold.
    """
    positions  = 40 * normal(key(seed), (DRONES, 3))
    velocities = 10 * normal(key(seed + 1), (DRONES, 3))
    sensed     = nearest(positions, NEIGHBORS)[-1]
    unsensed   = sorted(set(range(DRONES - 1)) - set(sensed.tolist()))
    reach      = float(sqrt(((positions[sensed] - positions[-1]) ** 2).sum(-1)).max())
    heading    = array(direction) / sqrt((array(direction) ** 2).sum())
    moved      = positions.at[unsensed[pick]].set(
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
            Consensus(centering_per_s=0.1),
            viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]])),
            [0.0, 1.1, 0.0],
            id = "consensus"
        ),
        param(
            Delayed(),
            viewed([[10.0, 0.0, 0.0]], delayed=array([[10.0, -1.0, 0.0]])),
            [0.0, -1.0, 0.0],
            id = "delayed"
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
            Overdamped(),
            viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]])),
            [0.0, 1.0, 0.0],
            id = "overdamped"
        ),
        param(
            Peripheral(blind_deg=90.0, sphere_m=0.2, weight_m_s2=12.0),
            viewed([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
            [2.0, 2.0, 0.0],
            id = "peripheral-at-the-border"
        ),
        param(
            Peripheral(blind_deg=90.0, sphere_m=0.2, weight_m_s2=12.0),
            viewed(
                [
                    [10.0, 0.0, 0.0],
                    [-10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, -10.0, 0.0]
                ]
            ),
            [0.0, 0.0, 0.0],
            id = "peripheral-inside"
        ),
        param(
            Peripheral(blind_deg=90.0, sphere_m=5.0, weight_m_s2=12.0),
            viewed([[10.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            [12.0 * 2**0.5 / 2, 0.0, 0.0],
            id = "peripheral-past-the-hard-sphere"
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
        param(White(amplitude=4.2), viewed([[10.0, 0.0, 0.0]]), (4.2, 0.0), id="white")
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


@mark.parametrize(
    ("position", "velocity", "expected"),
    [
        param((0.0, 0.0, 10.0), (-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), id="inside"),
        param((-120.0, 0.0, 10.0), (-10.0, 0.0, 0.0), (0.0, -2.0, 0.0), id="outward"),
        param((-120.0, 0.0, 10.0), (0.0, 10.0, 0.0), (1.0, 0.0, 0.0), id="along"),
        param((-120.0, 0.0, 10.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), id="inward"),
        param((0.0, 0.0, 30.0), (10.0, 0.0, 2.0), (0.0, 0.0, -1.6), id="above")
    ]
)
def test_the_roost_turns_a_drone_out_of_its_cylinder_back_and_pulls_it_level(
    expected : tuple[float, float, float],
    position : tuple[float, float, float],
    velocity : tuple[float, float, float]
):
    """
    Asserts that the roost turns a drone outside its cylinder across its
    heading toward the roost, at the full acceleration headed straight out,
    half of it headed along the edge, and none headed in, leaves one inside
    alone, and pulls one above the height down through its spring, damped at
    twice the square root of its stiffness.
    """
    roost = Roost(
        center_m         = (0.0, 0.0),
        height_m         = 10.0,
        radius_m         = 100.0,
        stiffness_per_s2 = 0.04,
        turning_m_s2     = 2.0
    )

    assert_allclose(
        roost(viewed([[1.0, 0.0, 0.0]], velocity, position=array(position))),
        expected,
        atol = 1e-6
    )


@mark.parametrize("factor", [2.0, 10.0, 30.0])
def test_the_roost_holds_the_heights_spread_in_proportion_to_the_flock_s_size(
    factor: float
):
    """
    Asserts that the roost and the white noise carried to twice a factor
    hold the stationary spread of the heights behind the plant's lag to
    twice the spread at that factor, within 10%, reading the stiffness and
    the damping off the roost's pull.
    """
    lag_s = Plant().lag_s

    def spread(scale: float) -> float:
        """
        Solves the stationary covariance of the height, the velocity, and
        the command under the roost's pull, the plant's lag, and the noise.
        """
        roost     = Roost().scaled(scale)
        height    = array([*roost.center_m, roost.height_m + 1.0])
        stiffness = -float(roost(viewed([[1.0, 0.0, 0.0]], position=height))[2])
        damping   = -float(roost(viewed([[1.0, 0.0, 0.0]], (10.0, 0.0, 1.0)))[2])
        drift     = [
            [0.0, 1.0, 0.0],
            [0.0, -1 / lag_s, 1 / lag_s],
            [-stiffness, 0.0, -damping]
        ]
        return solve_continuous_lyapunov(
            drift,
            -diag([0.0, 0.0, White().scaled(scale).amplitude ** 2])
        )[0, 0] ** 0.5

    assert_allclose(spread(2 * factor) / spread(factor), 2.0, rtol=0.1)


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
            Peripheral(blind_deg=90.0, sphere_m=0.5, weight_m_s2=8.0),
            Peripheral(blind_deg=90.0, sphere_m=2.0, weight_m_s2=2.0),
            id = "peripheral"
        ),
        param(
            Roost(stiffness_per_s2=32.0, turning_m_s2=8.0),
            Roost(stiffness_per_s2=2.0, turning_m_s2=2.0),
            id = "roost"
        ),
        param(White(amplitude=4.0), White(amplitude=2.0), id="white"),
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
    stiffness by its square, and its noise by its square root, leaving the
    roost's place and every slot holding a drone flock's own values.
    """
    assert_allclose(tree.leaves(component.scaled(4.0)), tree.leaves(expected))
