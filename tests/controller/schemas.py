"""
Pins the core's records, meaning the properties of a `Flock`, the neighbors
an `Observation` holds, and how a `Composite` sums its slots.
"""

from collections.abc import Callable
from common.drones   import flock_of, still_air, viewed
from dataclasses     import fields, replace
from jax             import tree, vmap
from jax.numpy       import array, eye, ones, ones_like, zeros
from lineax          import FunctionLinearOperator
from numpy.testing   import assert_allclose, assert_array_equal
from pytest          import mark, param, raises

from thermur.controller.components import (
    Consensus,
    Constant,
    InertialSpin,
    Marginal,
    Morse,
    Overdamped,
    Persistent,
    SoftCore,
    White
)
from thermur.controller.schemas import (
    Component,
    Composite,
    Noise,
    Observation,
    Speed,
    across,
    documented,
    nearest,
    torque,
    unit
)
from thermur.pipeline.settings  import Settings

EASTWARD = flock_of(zeros((8, 3)), array([10.0, 0.0, 0.0]) * ones((8, 3)))
"""
Eight drones stacked at the origin, each flying east at 10 m/s.
"""


def composite(**slots: Component) -> Composite:
    """
    Builds the composite the settings nest by default, carrying `slots`.
    """
    return replace(Composite.from_settings(Settings()), **slots)


@mark.parametrize(
    "act",
    [
        param(lambda view: Component()(view), id="component"),
        param(lambda view: Noise().amplitudes(view), id="noise"),
        param(lambda view: Speed().kicked(zeros(3), view), id="speed")
    ]
)
def test_a_bare_component_declares_no_action(act: Callable[[Observation], object]):
    """
    Asserts that the abstract slot, the abstract noise, and the abstract
    speed each raise until a subclass fills them.
    """
    with raises(NotImplementedError):
        act(viewed([[1.0, 0.0, 0.0]]))


def test_a_fixed_speed_leaves_no_acceleration_along_the_velocity():
    """
    Asserts that a speed slot holding the speed fixed takes out the part
    along the velocity that a free one and a heading leaving the speed free
    each add.
    """
    faster = viewed([[20.0, 0.0, 0.0]] * 7, neighbors=array([[12.0, 0.0, 0.0]] * 7))
    fixed  = composite(heading=Consensus(), speed=Marginal(fixed=True))(faster)[0][0]

    assert float(composite(heading=Consensus(), speed=Marginal())(faster)[0][0]) > 1.0
    assert_allclose(fixed, 0.0, atol=1e-4)


def test_a_flock_reads_its_polarization_at_each_extreme():
    """
    Asserts that a flock flying as one reads one and one whose headings
    cancel reads zero, whatever the speeds, on any leading axes.
    """
    together = flock_of(zeros((2, 4, 3)), array([1.0, 0.0, 0.0]) * ones((2, 4, 3)))
    against  = flock_of(
        zeros((2, 3)),
        array([[3.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    )

    assert_allclose(together.polarization, [1.0, 1.0])
    assert_allclose(against.polarization, 0.0)


def test_a_flock_reads_its_shape_from_the_gyration_tensor():
    """
    Asserts that drones on the axes at lengths one, two, and four give a
    gyration tensor holding each squared length over the count, and aspect
    ratios running from the thinnest axis.
    """
    lengths   = array([1.0, 2.0, 4.0])
    positions = array(
        [*(sign * length * row for length, row in zip(
            lengths,
            eye(3),
            strict = True
        ) for sign in (1, -1))]
    )
    flock = flock_of(positions, ones((6, 3)))

    assert_allclose(flock.gyration, array(eye(3)) * lengths**2 / 3, atol=1e-5)
    assert_allclose(flock.aspect_ratios, [1.0, 2.0, 4.0], rtol=1e-5)


def test_a_steering_heading_leaves_the_speed_to_the_speed_slot():
    """
    Asserts that a cohesion pushing along the velocity changes no speed
    under a steering heading or a heading leaving the speed free, so the
    acceleration along it is the speed slot's alone.
    """
    ahead = viewed([[20.0, 0.0, 0.0]] * 7, neighbors=array([[12.0, 0.0, 0.0]] * 7))
    slots = {
        "cohesion" : SoftCore(diameter_m=30.0),
        "contact"  : Morse(),
        "noise"    : White(),
        "speed"    : Marginal()
    }
    weighted = replace(ahead, weights=composite().coupling(ahead))
    pushed   = composite(**slots, heading=Overdamped())(ahead)[0][0]

    assert_allclose(pushed, Marginal()(weighted)[0], rtol=1e-5)
    assert_allclose(
        composite(**slots, heading=Consensus())(ahead)[0][0],
        Marginal()(weighted)[0] + Consensus()(weighted)[0],
        rtol = 1e-5
    )


def test_across_removes_the_part_along_each_velocity():
    """
    Asserts that `across` leaves each acceleration with no part along its
    drone's velocity and keeps the rest.
    """
    assert_allclose(
        across(
            array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]]),
            array([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        ),
        [[0.0, 4.0, 0.0], [1.0, 1.0, 0.0]],
        atol = 1e-6
    )


def test_cruising_keeps_each_heading_at_the_speed():
    """
    Asserts that `cruising` scales each velocity to the speed, holding each
    heading and the positions.
    """
    flock = flock_of(zeros((2, 3)), array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]))

    held = flock.cruising(10.0)

    assert_allclose(held.velocities, [[6.0, 8.0, 0.0], [0.0, 0.0, 10.0]])
    assert_array_equal(held.positions, flock.positions)


def test_diffusion_drives_the_velocity_and_the_force_and_never_a_position():
    """
    Asserts that the operator the Brownian path enters through scales the
    increment across each velocity by the noise's amplitude and the one
    along it by the speed slot's, scales the force's by the noise's, and
    reaches no position.
    """
    sensed = Observation.sensed(
        EASTWARD,
        EASTWARD.velocities,
        still_air(),
        0.0,
        7,
        10.0
    )
    increment = tree.map(ones_like, EASTWARD)
    slots     = {"heading": Overdamped(), "speed": Marginal(amplitude=0.5)}

    white = composite(**slots, noise=White(amplitude=2.0)).diffusion(
        0.0,
        EASTWARD,
        sensed
    )
    persistent = composite(
        **slots,
        noise = Persistent(persistence_s=0.5, strength_m_s2=3.0)
    ).diffusion(
        0.0, EASTWARD, sensed
    )

    kicked = white.mv(increment)
    forced = persistent.mv(increment)

    assert isinstance(white, FunctionLinearOperator)
    assert_allclose(kicked.positions, 0.0)
    assert_allclose(kicked.velocities, [[0.5, 2.0, 2.0]] * 8)
    assert_allclose(kicked.forces, 0.0)
    assert_allclose(kicked.spins, 0.0)
    assert_allclose(forced.velocities, [[0.5, 0.0, 0.0]] * 8)
    assert_allclose(forced.forces, 6.0)


def test_diffusion_under_a_spin_drives_the_spin_rather_than_the_heading():
    """
    Asserts that under the inertial spin the increment across the velocity
    reaches the spin as its torque, scaled by the square root of the
    friction over the inertia, and leaves the velocity only the increment
    along it.
    """
    kicked = composite(
        heading = InertialSpin(friction=0.25, inertia_s=2.0),
        noise   = White(amplitude=2.0),
        speed   = Marginal(amplitude=0.5)
    ).diffusion(
        0.0,
        EASTWARD,
        Observation.sensed(EASTWARD, EASTWARD.velocities, still_air(), 0.0, 7, 10.0)
    ).mv(
        flock_of(zeros((8, 3)), array([1.0, 0.0, 1.0]) * ones((8, 3)))
    )

    assert_allclose(kicked.velocities, [[0.5, 0.0, 0.0]] * 8)
    assert_allclose(kicked.spins, [[0.0, -0.5 * 2.0 / 10.0 / 2.0, 0.0]] * 8, rtol=1e-5)


def test_documented_sets_each_field_s_doc_to_its_docstring_unindented():
    """
    Asserts that each field of a documented component carries the docstring
    beneath it with its indentation and its surrounding newlines removed,
    and a field carrying none keeps no doc.
    """

    @documented
    class Spring(Component):
        """
        A stand-in component.
        """

        length_m: float = 1.0
        """
        The rest length, in meters, across
        two lines.
        """

        stiffness: float = 1.0

    assert {declared.name: declared.doc for declared in fields(Spring)} == {
        "length_m"  : "The rest length, in meters, across\ntwo lines.",
        "stiffness" : None
    }


def test_drift_moves_each_position_by_its_velocity():
    """
    Asserts that the vector field moves each position by its velocity,
    relaxes the force the noise holds, and sets each velocity's rate from
    the slots.
    """
    positions = array([[float(index) * 30.0, 0.0, 0.0] for index in range(8)])
    flock     = flock_of(positions, array([0.0, 10.0, 0.0]) * ones((8, 3)))
    sensed    = Observation.sensed(flock, flock.velocities, still_air(), 0.0, 7, 10.0)
    slots     = composite(noise=Persistent(persistence_s=2.0))

    rates               = slots.drift(0.0, flock, sensed)
    accelerations, spun = vmap(slots)(sensed)

    assert_array_equal(rates.positions, flock.velocities)
    assert_allclose(rates.forces, 0.0)
    assert_allclose(rates.velocities, accelerations)
    assert_allclose(rates.spins, spun)


def test_from_settings_reads_each_slot_by_name():
    """
    Asserts that each slot of the composite is the settings field naming it.
    """
    settings = Settings(coupling=Constant(coupling_per_s=2.0))

    built = Composite.from_settings(settings)

    assert built.coupling == settings.coupling
    assert built.cohesion == settings.cohesion


def test_nearest_leaves_a_drone_out_of_its_own_row_at_a_shared_position():
    """
    Asserts that drones stacked at one point each list every other drone,
    which `KDTree.query` may order ahead of the drone itself.
    """
    for own, row in enumerate(nearest(zeros((4, 3)), 3)):
        assert sorted(row) == sorted({0, 1, 2, 3} - {own})


def test_nearest_orders_neighbors_by_distance_over_a_batch():
    """
    Asserts that a batch of flocks keeps its leading axis, with each drone's
    row running from its nearest neighbor outward.
    """
    line = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])

    rows = nearest(array([line, line[::-1]]), 2)

    assert rows.shape == (2, 4, 2)
    assert rows[0, 0].tolist() == [1, 2]
    assert rows[1, 0].tolist() == [1, 2]
    assert rows[0, 3].tolist() == [2, 1]


def test_sensing_holds_each_neighbor_s_state_by_rank():
    """
    Asserts that an observation holds each neighbor's offset, distance,
    velocity and delayed velocity by rank, beside the air at the drone's
    position, the cruise speed, and the drone's own spin.
    """
    positions  = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    velocities = array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    flock      = flock_of(positions, velocities)

    sensed = Observation.sensed(flock, 2 * velocities, still_air(5.0), 0.0, 2, 9.0)

    assert_allclose(sensed.offsets[0], [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [1.0, 3.0])
    assert_allclose(sensed.neighbors[0], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert_allclose(sensed.delayed[0], [[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert_allclose(sensed.air["excess"], 5.0)
    assert_allclose(sensed.cruise, [9.0, 9.0, 9.0])
    assert_array_equal(sensed.spin, flock.spins)


def test_summing_a_neighbor_on_the_drone_stays_finite():
    """
    Asserts that a neighbor at the drone's exact position takes a bearing of
    zero, so a slot reading it stays finite.
    """
    assert_allclose(viewed([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).bearings[0], 0.0)


@mark.parametrize("rate", [0.0, 1.0, 2.0])
def test_the_heading_turns_a_drone_at_the_rate_its_coupling_sets(rate: float):
    """
    Asserts that the composite hands its heading the rate the coupling sets,
    so a drone whose neighbor flies one meter per second across it turns at
    that rate times the difference, and the inertial spin takes that pull as
    a torque over its inertia.
    """
    view     = viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]]))
    coupling = Constant(coupling_per_s=rate)
    fixed    = Marginal(fixed=True)

    assert_allclose(
        composite(coupling=coupling, heading=Overdamped(), speed=fixed)(view)[0],
        [0.0, rate, 0.0],
        atol = 1e-6
    )
    assert_allclose(
        composite(coupling=coupling, heading=InertialSpin(inertia_s=2.0))(view)[1],
        [0.0, 0.0, rate / 10.0 / 2.0],
        atol = 1e-6
    )


@mark.parametrize(
    ("acceleration", "velocity", "expected"),
    [
        ([0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ]
)
def test_torque_turns_a_velocity_at_its_rate_and_holds_a_stopped_drone(
    acceleration : list[float],
    velocity     : list[float],
    expected     : list[float]
):
    """
    Asserts that `torque` returns the angular velocity at which an
    acceleration across a velocity turns it, and none for a stopped drone.
    """
    assert_allclose(torque(array(acceleration), array(velocity)), expected)


@mark.parametrize(
    ("vectors", "expected"),
    [
        ([3.0, 4.0, 0.0], [0.6, 0.8, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ]
)
def test_unit_divides_by_length_and_holds_a_zero_vector(
    vectors  : list[float],
    expected : list[float]
):
    """
    Asserts that `unit` returns a vector of length one, or zero for zero.
    """
    assert_allclose(unit(array(vectors)), expected)
