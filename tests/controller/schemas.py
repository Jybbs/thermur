"""
Pins the core's records, meaning the properties of a `Flock`, what the
`Radio` delivers and loses, the neighbors an `Observation` holds, and how a
`Composite` sums its slots.
"""

from collections.abc import Callable
from common.drones   import flock_of, still_air, viewed
from dataclasses     import fields, replace
from jax             import Array, jit, lax, tree, vmap
from jax.numpy       import (
    arange,
    array,
    broadcast_to,
    eye,
    full,
    isfinite,
    linalg,
    newaxis,
    ones,
    ones_like,
    sort,
    stack,
    where,
    zeros
)
from jax.random    import key, normal, uniform
from lineax        import FunctionLinearOperator
from math          import inf
from numpy.testing import assert_allclose, assert_array_equal
from pytest        import mark, param, raises

from thermur.controller.components import (
    Centroid,
    Consensus,
    Constant,
    Cruise,
    HardSphere,
    InertialSpin,
    Marginal,
    Overdamped,
    Persistent,
    White,
    Zig
)
from thermur.controller.schemas import (
    Component,
    Composite,
    Inbox,
    Noise,
    Observation,
    Radio,
    across,
    nearest,
    torque,
    unit,
    zorder
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


def holds(inbox: Inbox, receiver: int, sender: int) -> bool:
    """
    Finds whether a slot of the receiver's table hears the sender.
    """
    return bool((inbox.heard[receiver] & (inbox.senders[receiver] == sender)).any())


@mark.parametrize(
    "act",
    [
        param(lambda view: Component()(view), id="component"),
        param(lambda view: Noise().amplitudes(view), id="noise")
    ]
)
def test_a_bare_component_declares_no_action(act: Callable[[Observation], object]):
    """
    Asserts that the abstract slot and the abstract noise each raise until a
    subclass fills them.
    """
    with raises(NotImplementedError):
        act(viewed([[1.0, 0.0, 0.0]]))


def test_a_drone_learns_a_sender_beyond_its_table_from_a_member_s_list():
    """
    Asserts that a drone hears a sender its table does not hold once a
    member it hears lists that sender, as two-hop sensing learns one, and
    does not hear a nearer sender no member lists and the curve does not set
    beside it.
    """
    positions = array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
    )
    flock = flock_of(positions, zeros((4, 3)))
    radio = Radio(
        delay_s = 0.0,
        loss    = 0.0,
        range_m = 100.0,
        rate_hz = 10.0
    )
    _, sent = radio.opened(flock, 0, 0.1)
    table   = Inbox.from_broadcast(flock, radio.range_m, flock, 2)
    listed  = table.senders.at[1].set(array([3, 3]))
    inbox   = replace(
        table,
        heard   = table.heard.at[0].set(array([True, False])),
        lists   = listed[newaxis],
        senders = listed.at[0].set(array([1, 0]))
    )

    heard, _ = radio.heard(flock, inbox, key(0), 0, sent, 1, 0.1)

    assert holds(heard, 0, 3)
    assert not holds(heard, 0, 2)


def test_a_fixed_speed_leaves_no_acceleration_along_the_velocity():
    """
    Asserts that a speed slot holding the speed fixed takes out the part
    along the velocity that a free one and a heading leaving the speed free
    each add.
    """
    faster = viewed([[20.0, 0.0, 0.0]] * 7, neighbors=array([[20.0, 0.0, 0.0]] * 7))
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


def test_a_lone_drone_hears_no_sender_and_senses_only_itself():
    """
    Asserts that a flock of one drone keeps an empty table through its
    packets, and that sensing then reads every rank as a neighbor sitting on
    the drone.
    """
    radio       = Radio(delay_s=0.0, loss=0.0, rate_hz=10.0)
    flock       = flock_of(zeros((1, 3)), ones((1, 3)))
    inbox, sent = radio.opened(flock, 0, 0.1)
    for step in range(1, 4):
        inbox, sent = radio.heard(flock, inbox, key(0), 0, sent, step, 0.1)

    assert not inbox.heard.any()
    assert_allclose(
        Observation.sensed(still_air(), 7, 10.0, flock, inbox, 0.0).offsets,
        0.0
    )


def test_a_noise_holding_no_state_leaves_it_as_it_stands():
    """
    Asserts that a noise slot whose state no step updates returns the state
    the drone holds, whatever the draw and the roost's turn.
    """
    observation = viewed([[1.0, 0.0, 0.0]], force=array([1.0, 2.0, 3.0]))

    assert_array_equal(
        White().excited(0.5, array(0.0), observation, 0.1, array([9.0, 9.0, 0.0])),
        [1.0, 2.0, 3.0]
    )


def test_a_radio_faster_than_the_step_delivers_at_every_step():
    """
    Asserts that a radio sending more packets a second than the integrator
    takes steps delivers one packet at every step rather than none.
    """
    radio       = Radio(delay_s=0.0, loss=0.0, rate_hz=50.0)
    inbox, sent = radio.opened(flock_of(zeros((3, 3)), zeros((3, 3))), 0, 0.1)

    heard, _ = radio.heard(
        flock_of(ones((3, 3)), ones((3, 3))),
        inbox,
        key(0),
        0,
        sent,
        3,
        0.1
    )

    assert radio.period_steps(0.1) == 1
    assert_allclose(heard.positions[heard.heard], 1.0)


def test_a_steering_heading_leaves_the_speed_to_the_speed_slot():
    """
    Asserts that a cohesion and a contact pushing along the velocity change
    no speed under a steering heading or a heading leaving the speed free,
    so the acceleration along it is the speed slot's alone.
    """
    ahead = viewed([[20.0, 0.0, 0.0]] * 7, neighbors=array([[12.0, 0.0, 0.0]] * 7))
    slots = {
        "cohesion" : Centroid(),
        "contact"  : HardSphere(sphere_m=30.0),
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


def test_cruising_keeps_each_command_s_direction_at_the_speed():
    """
    Asserts that `cruising` scales each command to the speed, holding each
    command's direction, the velocities, and the positions.
    """
    flock = flock_of(
        zeros((2, 3)),
        ones((2, 3)),
        array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    )

    held = flock.cruising(10.0)

    assert_allclose(held.commands, [[6.0, 8.0, 0.0], [0.0, 0.0, 10.0]])
    assert_array_equal(held.velocities, flock.velocities)
    assert_array_equal(held.positions, flock.positions)


def test_diffusion_drives_the_command_and_the_force_and_never_a_position():
    """
    Asserts that the operator the Brownian path enters through scales the
    increment across each command by the noise's amplitude and the one along
    it by the speed slot's, scales the force's by the noise's, and reaches
    no position or velocity.
    """
    sensed = Observation.sensed(
        still_air(),
        7,
        10.0,
        EASTWARD,
        Inbox.from_broadcast(EASTWARD, inf, EASTWARD),
        0.0
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
    assert_allclose(kicked.velocities, 0.0)
    assert_allclose(kicked.commands, [[0.5, 2.0, 2.0]] * 8)
    assert_allclose(kicked.forces, 0.0)
    assert_allclose(kicked.spins, 0.0)
    assert_allclose(forced.commands, [[0.5, 0.0, 0.0]] * 8)
    assert_allclose(forced.forces, 6.0)


def test_diffusion_under_a_spin_drives_the_spin_rather_than_the_heading():
    """
    Asserts that under the inertial spin the increment across the command
    reaches the spin as its torque, scaled by the square root of the
    friction over the inertia, and leaves the command only the increment
    along it.
    """
    kicked = composite(
        heading = InertialSpin(friction=0.25, inertia_s=2.0),
        noise   = White(amplitude=2.0),
        speed   = Marginal(amplitude=0.5)
    ).diffusion(
        0.0,
        EASTWARD,
        Observation.sensed(
            still_air(),
            7,
            10.0,
            EASTWARD,
            Inbox.from_broadcast(EASTWARD, inf, EASTWARD),
            0.0
        )
    ).mv(
        flock_of(zeros((8, 3)), array([1.0, 0.0, 1.0]) * ones((8, 3)))
    )

    assert_allclose(kicked.commands, [[0.5, 0.0, 0.0]] * 8)
    assert_allclose(kicked.spins, [[0.0, -0.5 * 2.0 / 10.0 / 2.0, 0.0]] * 8, rtol=1e-5)


def test_drift_moves_each_position_by_its_velocity_and_the_wind():
    """
    Asserts that the vector field moves each position by its velocity plus
    the wind, relaxes the force the noise holds, sets each command's rate
    from the slots, and carries each velocity toward its command through
    the plant.
    """
    flock = flock_of(
        array([[float(index) * 30.0, 0.0, 0.0] for index in range(8)]),
        array([0.0, 10.0, 0.0]) * ones((8, 3)),
        array([0.0, 12.0, 0.0]) * ones((8, 3))
    )
    sensed = Observation.sensed(
        still_air(wind=(1.0, -2.0, 0.5)),
        7,
        12.0,
        flock,
        Inbox.from_broadcast(flock, inf, flock),
        0.0
    )
    slots = composite(noise=Persistent(persistence_s=2.0))

    rates               = slots.drift(0.0, flock, sensed)
    accelerations, spun = vmap(slots)(sensed)

    assert_allclose(rates.positions, flock.velocities + array([1.0, -2.0, 0.5]))
    assert_allclose(rates.forces, 0.0)
    assert_allclose(rates.commands, accelerations)
    assert_allclose(rates.spins, spun)
    assert_allclose(rates.velocities, [[0.0, 2.0, 0.0]] * 8, rtol=1e-6)


def test_exciting_the_flock_copies_a_zig_sooner_by_the_radio_s_delay():
    """
    Asserts that a resting drone hearing two zigging neighbors, under
    a branching that sets it off for certain, starts its zig after its
    reaction time and the cue's less the radio's delay, while each zigging
    neighbor counts down by the step.
    """
    built = composite(
        noise = Zig(
            accel_m_s2   = 2.0,
            back_s       = 1.0,
            branching    = 2.0,
            cue_s        = 0.5,
            reaction_s   = 1.5,
            refractory_s = 3.0,
            side_s       = 1.0
        ),
        radio = replace(Composite.from_settings(Settings()).radio, delay_s=0.5)
    )
    flock = replace(
        flock_of(
            array(
                [
                    [1000.0, 1000.0, 100.0],
                    [1000.0, 1005.0, 100.0],
                    [1000.0, 995.0, 100.0]
                ]
            ),
            array([[10.0, 0.0, 0.0]] * 3)
        ),
        forces = array([[0.0, 0.0, 0.0], [4.5, 1.0, 0.0], [4.5, 1.0, 0.0]])
    )

    excited = built.excited(
        flock,
        key(0),
        Observation.sensed(
            still_air(),
            2,
            10.0,
            flock,
            Inbox.from_broadcast(flock, built.radio.range_m, flock),
            0.0
        ),
        0.1
    )

    assert_allclose(
        excited.forces,
        [[3.0 + 2.0 + 1.5 + 0.5 - 0.5, 1.0, 0.0], [4.4, 1.0, 0.0], [4.4, 1.0, 0.0]],
        rtol = 1e-6
    )


def test_exciting_the_flock_counts_down_and_starts_a_zig_at_the_rim():
    """
    Asserts that one step of the default composite's noise leaves a resting
    drone over the roost at rest, counts a zig down by the step, and starts
    a zig on a drone the roost turns hard past its rim, toward the turn.
    """
    built = composite()
    flock = replace(
        flock_of(
            array(
                [
                    [1000.0, 1000.0, 100.0],
                    [1000.0, 1005.0, 100.0],
                    [2500.0, 1000.0, 100.0]
                ]
            ),
            array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        ),
        forces = array([[0.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    )
    zig = built.noise

    excited = built.excited(
        flock,
        key(0),
        Observation.sensed(
            still_air(),
            2,
            10.0,
            flock,
            Inbox.from_broadcast(flock, built.radio.range_m, flock),
            0.0
        ),
        0.1
    )

    assert_allclose(
        excited.forces,
        [
            [0.0, 0.0, 0.0],
            [4.9, 1.0, 0.0],
            [zig.refractory_s + zig.side_s + zig.back_s + zig.reaction_s, 1.0, 0.0]
        ],
        rtol = 1e-6
    )


def test_from_settings_reads_each_slot_by_name_scaled_to_the_launch():
    """
    Asserts that each slot of the composite is the settings field naming it,
    carried from a starling flock to the launch by the plan's similarity.
    """
    settings = Settings(coupling=Constant(coupling_per_s=2.0))

    built = Composite.from_settings(settings)

    assert {slot.name: getattr(built, slot.name) for slot in fields(Composite)} == {
        slot.name: getattr(settings, slot.name).scaled(settings.plan.similarity)
        for slot in fields(Composite)
    }


def test_nearest_leaves_a_drone_out_of_its_own_row_at_a_shared_position():
    """
    Asserts that drones stacked at one point each list every other drone,
    which `KDTree.query` may order ahead of the drone itself.
    """
    for own, row in enumerate(nearest(3, zeros((4, 3)))):
        assert sorted(row) == sorted({0, 1, 2, 3} - {own})


@mark.parametrize("rate", [0.0, 1.0, 2.0])
def test_the_heading_turns_a_drone_at_the_rate_its_coupling_sets(rate: float):
    """
    Asserts that the composite hands its heading the rate the coupling sets,
    so a drone whose neighbor flies one meter per second across it turns at
    that rate times the difference under the heading holding the speed, and
    the inertial spin takes that pull as a torque over its inertia.
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


def test_nearest_orders_neighbors_by_distance_over_a_batch():
    """
    Asserts that a batch of flocks keeps its leading axis, with each drone's
    row running from its nearest neighbor outward.
    """
    line = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])

    rows = nearest(2, array([line, line[::-1]]))

    assert rows.shape == (2, 4, 2)
    assert rows[0, 0].tolist() == [1, 2]
    assert rows[1, 0].tolist() == [1, 2]
    assert rows[0, 3].tolist() == [2, 1]


def test_sensing_holds_each_neighbor_s_heard_state_by_rank():
    """
    Asserts that an observation ranks the neighbors by the distance the
    drone heard and holds each one's offset, distance, velocity and delayed
    velocity as the drone heard them, beside the drone's own position,
    command, and spin, the cruise speed, and the air at its position.
    """
    positions  = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    velocities = array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    flock      = flock_of(positions, velocities, 5 * velocities)
    sent       = flock_of(positions - array([0.5, 0.0, 0.0]), 3 * velocities)

    sensed = Observation.sensed(
        still_air(5.0),
        2,
        10.0,
        flock,
        Inbox.from_broadcast(flock_of(positions, 2 * velocities), inf, sent),
        0.0
    )

    assert_allclose(sensed.offsets[0], [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [0.5, 2.5])
    assert_allclose(sensed.neighbors[0], [[6.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    assert_allclose(sensed.delayed[0], [[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert_array_equal(sensed.position, positions)
    assert_array_equal(sensed.velocity, flock.commands)
    assert_array_equal(sensed.spin, flock.spins)
    assert_allclose(sensed.air["excess"], 5.0)
    assert_allclose(sensed.cruise, [10.0, 10.0, 10.0])


@mark.parametrize(
    ("slot", "idle"),
    [
        param("roost", Centroid(stiffness_per_s2=0.0), id="roost"),
        param("speed", Cruise(cruise_m_s=8.0), id="speed")
    ]
)
def test_the_composite_adds_each_slot_holding_the_flock_to_its_sum(
    slot : str,
    idle : Component
):
    """
    Asserts that the composite's command rate carries the roost's turn and
    pull and the speed slot's acceleration beside every other slot, against
    a composite holding that slot idle, for a drone slower than the cruise
    speed, high, and past the rim of the roost.
    """
    view = viewed(
        [[20.0, 0.0, 0.0]] * 7,
        velocity = (8.0, 0.0, 0.0),
        cruise   = array(10.0),
        position = array([-10.0, 1700.0, 130.0])
    )
    settings = Settings()

    assert_allclose(
        composite()(view)[0] - composite(**{slot: idle})(view)[0],
        getattr(settings, slot).scaled(settings.plan.similarity)(view),
        atol = 1e-5
    )


def test_sensing_ranks_neighbors_by_the_positions_the_drone_heard():
    """
    Asserts that a drone ranks its neighbors by where it heard each one
    rather than where each one now flies, so a neighbor heard nearer ranks
    first and keeps the distance the drone heard.
    """
    positions = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    flock     = flock_of(positions, ones((3, 3)))
    sent      = flock_of(
        array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ones((3, 3))
    )

    sensed = Observation.sensed(
        still_air(),
        2,
        10.0,
        flock,
        Inbox.from_broadcast(flock, inf, sent),
        0.0
    )

    assert_allclose(sensed.offsets[0], [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [2.0, 4.0])


def test_sensing_reads_a_neighbor_out_of_range_as_one_on_the_drone():
    """
    Asserts that a rank no sender heard within the range fills reads as a
    neighbor at the drone's own position flying its command, so the drone
    alone in range adds nothing to a sum over its neighbors.
    """
    positions = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    flock     = flock_of(positions, ones((3, 3)), 2 * ones((3, 3)))

    sensed = Observation.sensed(
        still_air(),
        2,
        10.0,
        flock,
        Inbox.from_broadcast(flock, 10.0, flock),
        0.0
    )

    assert_allclose(sensed.offsets[0], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert_allclose(sensed.offsets[2], 0.0)
    assert_allclose(sensed.neighbors[0], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert_allclose(sensed.delayed[2], 2.0)


def test_sensing_reads_centrality_over_every_held_sender():
    """
    Asserts that each drone's centrality is the length of the mean bearing
    to every sender it holds, past its nearest neighbors, zero for a drone
    holding none, and that each neighbor's noise state is held by rank.
    """
    flock = replace(
        flock_of(
            array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 100.0]
                ]
            ),
            ones((5, 3))
        ),
        forces = array([[float(drone), 0.0, 0.0] for drone in range(5)])
    )

    sensed = Observation.sensed(
        still_air(),
        1,
        10.0,
        flock,
        Inbox.from_broadcast(flock, 10.0, flock),
        0.0
    )

    assert_allclose(sensed.centrality[0], 1 / 3, rtol=1e-6)
    assert_allclose(sensed.centrality[4], 0.0)
    assert_allclose(sensed.forces[0], [[1.0, 0.0, 0.0]])


def test_summing_a_neighbor_on_the_drone_stays_finite():
    """
    Asserts that a neighbor at the drone's exact position takes a bearing of
    zero, so a slot reading it stays finite.
    """
    assert_allclose(viewed([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).bearings[0], 0.0)


def test_the_curve_order_follows_the_drones_along_a_line():
    """
    Asserts that drones spread along one axis take their order along it on
    the Z-order curve.
    """
    offsets_m = array([5.0, 1.0, 4.0, 2.0, 3.0])

    assert zorder(
        stack([offsets_m, 2 * offsets_m, zeros(5)], 1)
    ).tolist() == [1, 3, 4, 2, 0]


def test_the_radio_carried_to_a_larger_flock_stays_a_drone_radio():
    """
    Asserts that the radio carried to a flock four times a starling flock's
    size keeps the delay, the rate, the loss, and the range a drone's radio
    sets.
    """
    radio = Radio(
        delay_s = 0.5,
        loss    = 0.2,
        range_m = 100.0,
        rate_hz = 10.0
    )

    assert radio.scaled(4.0) == radio


def test_the_radio_delivers_each_broadcast_its_delay_late_at_each_packet():
    """
    Asserts that the radio delivers, at each packet, the broadcast sent
    `delay_s` earlier and the velocity the lag before it, holds what each
    drone heard between packets, and opens with the launch heard by every
    drone and kept as every broadcast.
    """
    radio  = Radio(delay_s=0.3, loss=0.0, rate_hz=5.0)
    flocks = [
        flock_of(full((3, 3), step), full((3, 3), 10.0 * step))
        for step in range(8)
    ]
    inbox, sent = radio.opened(flocks[0], 1, 0.1)
    heard       = {}
    for step in range(1, 8):
        inbox, sent = radio.heard(flocks[step], inbox, key(0), 1, sent, step, 0.1)
        heard[step] = inbox

    assert sent.positions.shape == (5, 3, 3)
    assert_allclose(heard[2].positions[heard[2].heard], 0.0)
    assert_allclose(heard[4].positions[heard[4].heard], 1.0)
    assert_allclose(heard[5].positions[heard[5].heard], 1.0)
    assert_allclose(heard[6].positions[heard[6].heard], 3.0)
    assert_allclose(heard[6].velocities[heard[6].heard], 30.0)
    assert_allclose(heard[6].delayed[heard[6].heard], 20.0)
    assert_allclose(heard[7].positions[heard[7].heard], 3.0)


def test_the_radio_draws_each_packet_s_loss_apart_at_each_step():
    """
    Asserts that each step draws its own losses, so a packet lost at two
    steps in a row falls at the square of the loss rather than at the loss.
    """
    radio = Radio(loss=0.4)
    lost  = [
        radio.draws(key(0), arange(200)[:, newaxis], arange(200), step) >= 0.6
        for step in (1, 2)
    ]

    assert_allclose(float((lost[0] & lost[1]).mean()), 0.16, atol=0.01)


def test_the_radio_forgets_a_sender_once_its_hold_passes_with_no_packet():
    """
    Asserts that a drone keeps hearing a sender whose packets stop until
    `hold_periods` packet periods pass from its last packet, then forgets
    it, and hears it again at its next packet.
    """
    radio = Radio(
        delay_s      = 0.0,
        hold_periods = 3,
        loss         = 0.0,
        range_m      = 10.0,
        rate_hz      = 5.0
    )
    near        = flock_of(array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), zeros((2, 3)))
    far         = flock_of(array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), zeros((2, 3)))
    inbox, sent = radio.opened(near, 0, 0.1)
    heard       = []
    for step, flock in enumerate([far] * 7 + [near], start=1):
        inbox, sent = radio.heard(flock, inbox, key(0), 0, sent, step, 0.1)
        heard.append(holds(inbox, 0, 1))

    assert heard == [True] * 5 + [False] * 2 + [True]


def test_the_radio_hears_at_launch_only_the_drones_within_its_range():
    """
    Asserts that the inbox a flight opens on holds each drone within the
    range and none past it, so sensing pads a rank no heard sender fills.
    """
    flock = flock_of(
        array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
        ones((3, 3))
    )
    inbox, _ = Radio(range_m=10.0).opened(flock, 0, 0.1)

    assert_array_equal(inbox.senders[0][inbox.heard[0]], [1])
    assert_allclose(
        Observation.sensed(still_air(), 2, 10.0, flock, inbox, 0.0).offsets[0],
        [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )


@mark.parametrize("drones", [param(20, id="twenty"), param(200, id="two-hundred")])
def test_the_radio_opens_on_a_table_of_the_nearest_senders_whatever_the_flock_s_size(
    drones: int
):
    """
    Asserts that each drone's table holds the radio's `slots` senders, the
    nearest drones to it at launch, at any flock size.
    """
    positions = uniform(key(drones), (drones, 3)) * 30.0
    inbox, _ = Radio(range_m=inf).opened(flock_of(positions, ones((drones, 3))), 0, 0.1)
    held     = min(Radio.slots, drones - 1)

    assert inbox.senders.shape == inbox.heard.shape == (drones, Radio.slots)
    assert inbox.heard.sum(1).tolist() == [held] * drones
    assert_array_equal(
        sort(where(inbox.heard, inbox.senders, drones), axis=1)[:, :held],
        sort(nearest(held, positions), axis=1)
    )


def test_the_radio_loses_its_share_of_the_packets_between_two_drones():
    """
    Asserts that each drone hears each other drone's packet with the chance
    the loss leaves, each pair drawn apart, and that a packet lost leaves
    what the drone heard before.
    """
    radio = Radio(delay_s=0.0, loss=0.4, rate_hz=10.0)
    draws = radio.draws(key(0), arange(200)[:, newaxis], arange(200), 1)
    lost  = next(
        step
        for step in range(1, 50)
        if radio.draws(key(0), array([[0]]), array([[1]]), step)[0, 0] >= 0.6
    )
    inbox, sent = radio.opened(flock_of(zeros((2, 3)), zeros((2, 3))), 0, 0.1)

    inbox, _ = radio.heard(
        flock_of(ones((2, 3)), ones((2, 3))),
        inbox,
        key(0),
        0,
        sent,
        lost,
        0.1
    )

    assert_allclose(float((draws >= 0.6).mean()), 0.4, atol=0.01)
    assert holds(inbox, 0, 1)
    assert_allclose(inbox.positions[0, 0], 0.0)


def test_the_radio_s_range_acts_on_the_distance_as_each_packet_is_sent():
    """
    Asserts that a packet sent from past the range never arrives, leaving
    what the drone heard before, while one sent from inside it arrives even
    where the sender has since flown past the range.
    """
    radio = Radio(
        delay_s = 0.1,
        loss    = 0.0,
        range_m = 10.0,
        rate_hz = 10.0
    )
    near        = flock_of(array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), zeros((2, 3)))
    far         = flock_of(array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), zeros((2, 3)))
    inbox, sent = radio.opened(near, 0, 0.1)

    inbox, sent = radio.heard(far, inbox, key(0), 0, sent, 1, 0.1)
    assert_allclose(inbox.positions[0, 0], [5.0, 0.0, 0.0])

    inbox, sent = radio.heard(far, inbox, key(0), 0, sent, 2, 0.1)
    assert_allclose(inbox.positions[0, 0], [5.0, 0.0, 0.0])
    assert holds(inbox, 0, 1)


@mark.parametrize(
    "drones",
    [
        param(100, id="hundred"),
        param(1000, id="thousand", marks=mark.slow),
        param(3000, id="three-thousand", marks=mark.slow)
    ]
)
def test_the_neighbors_agree_with_an_exact_search_over_the_same_broadcasts(drones: int):
    """
    Asserts that the plan's `neighbors` nearest senders each drone's table
    yields match, for at least 99% of the neighbors, those an exact search
    finds over every broadcast the drone took, each lost where the radio's
    own draw for the pair loses it, after 12 s of a flock at the plan's
    spacing whose velocities scatter by 1.5 m/s about 10 m/s east.
    """
    radio       = Radio()
    positions   = uniform(key(drones), (drones, 3)) * 15.0 * drones ** (1 / 3)
    velocities  = array([10.0, 0.0, 0.0]) + 1.5 * normal(key(drones + 1), (drones, 3))
    rows        = arange(drones)[:, newaxis]
    hold        = radio.hold_periods * radio.period_steps(0.1)
    inbox, sent = radio.opened(flock_of(positions, velocities), 0, 0.1)
    exact       = (
        linalg.norm(positions[:, newaxis] - positions, axis=-1) <= radio.range_m
    ) & ~eye(drones, dtype=bool)
    told     = broadcast_to(positions, (drones, drones, 3))
    received = zeros((drones, drones), dtype="int32")

    @jit
    def searched(
        exact    : Array,
        received : Array,
        sent_at  : Array,
        step     : int,
        told     : Array
    ) -> tuple[Array, Array, Array]:
        """
        Delivers one packet to every pair of drones within range.
        """
        carried = (
            (linalg.norm(sent_at[:, newaxis] - sent_at, axis=-1) <= radio.range_m)
            & (radio.draws(key(0), rows, arange(drones), step) < 1 - radio.loss)
            & ~eye(drones, dtype=bool)
        )
        return (
            carried | exact & (step - received < hold),
            where(carried, step, received),
            where(carried[..., newaxis], sent_at, told)
        )

    for step in range(1, 121):
        inbox, sent = radio.heard(
            flock_of(positions + step * 0.1 * velocities, velocities),
            inbox,
            key(0),
            0,
            sent,
            array(step),
            0.1
        )
        if step % radio.period_steps(0.1) == 0:
            exact, received, told = searched(
                exact,
                received,
                positions + max(step - radio.delay_steps(0.1), 0) * 0.1 * velocities,
                step,
                told
            )

    now           = positions + 12.0 * velocities
    tabled, slots = lax.top_k(
        -where(
            inbox.heard,
            linalg.norm(inbox.positions - now[:, newaxis], axis=-1),
            inf
        ),
        Settings().plan.neighbors
    )
    searched_m, found = lax.top_k(
        -where(exact, linalg.norm(told - now[:, newaxis], axis=-1), inf),
        Settings().plan.neighbors
    )
    matched = (
        where(isfinite(tabled), inbox.senders[rows, slots], -1)[..., newaxis]
        == where(isfinite(searched_m), found, -2)[:, newaxis]
    ).any(-1)

    assert float(matched.sum() / isfinite(searched_m).sum()) >= 0.99
