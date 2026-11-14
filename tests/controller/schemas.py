"""
Pins the core's records, meaning the properties of a `Flock`, what the
`Radio` delivers and loses, the neighbors an `Observation` holds, and how a
`Composite` sums its slots.
"""

from common.drones import flock_of, still_air, viewed
from dataclasses   import fields, replace
from jax           import vmap
from jax.numpy     import array, eye, full, ones, zeros
from jax.random    import key
from lineax        import DiagonalLinearOperator
from math          import inf
from numpy.testing import assert_allclose, assert_array_equal
from pytest        import mark, raises

from thermur.controller.components import (
    Centroid,
    Consensus,
    Constant,
    HardSphere,
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
    unit
)
from thermur.pipeline.settings  import Settings


def composite(**slots: Component) -> Composite:
    """
    Builds the composite the settings nest by default, carrying `slots`.
    """
    return replace(Composite.from_settings(Settings()), **slots)


def test_a_bare_component_declares_no_action():
    """
    Asserts that the abstract slot and the abstract noise each raise until a
    subclass fills them.
    """
    with raises(NotImplementedError):
        Component()(viewed([[1.0, 0.0, 0.0]]))

    with raises(NotImplementedError):
        Noise().amplitudes(viewed([[1.0, 0.0, 0.0]]))


def test_a_confined_heading_accelerates_only_across_the_velocity():
    """
    Asserts that a slot pushing along the velocity turns no drone under a
    heading holding the speed, and reaches it under one leaving the speed
    free.
    """
    ahead = viewed([[20.0, 0.0, 0.0]] * 7)
    slots = {
        "cohesion" : Centroid(),
        "noise"    : White(),
        "contact"  : HardSphere(sphere_m=30.0)
    }

    assert_allclose(composite(heading=Overdamped(), **slots)(ahead)[0], 0.0, atol=1e-4)
    assert abs(float(composite(heading=Consensus(), **slots)(ahead)[0])) > 1.0


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
    radio = Radio(delay_s=0.0, loss=0.0, rate_hz=50.0)

    assert radio.period_steps(0.1) == 1
    assert_allclose(
        radio.heard(
            *radio.opened(flock_of(zeros((3, 3)), zeros((3, 3))), 0, 0.1),
            flock_of(ones((3, 3)), ones((3, 3))),
            key(0),
            3,
            0,
            0.1
        )[0].positions,
        1.0
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
    Asserts that the operator the Brownian path enters through holds zero on
    each position and each velocity and each amplitude the noise gives.
    """
    flock  = flock_of(zeros((8, 3)), ones((8, 3)))
    sensed = Observation.sensed(
        flock,
        Inbox.from_broadcast(flock, inf, flock),
        still_air(),
        0.0,
        7
    )

    white      = composite(noise=White(amplitude=2.0)).diffusion(0.0, flock, sensed)
    persistent = composite(
        noise = Persistent(persistence_s=0.5, strength_m_s2=3.0)
    ).diffusion(
        0.0, flock, sensed
    )

    assert isinstance(white, DiagonalLinearOperator)
    assert_allclose(white.diagonal.positions, 0.0)
    assert_allclose(white.diagonal.velocities, 0.0)
    assert_allclose(white.diagonal.commands, 2.0)
    assert_allclose(white.diagonal.forces, 0.0)
    assert_allclose(persistent.diagonal.commands, 0.0)
    assert_allclose(persistent.diagonal.forces, 6.0)


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
        flock,
        Inbox.from_broadcast(flock, inf, flock),
        still_air(wind=(1.0, -2.0, 0.5)),
        0.0,
        7
    )
    slots = composite(noise=Persistent(persistence_s=2.0))

    rates = slots.drift(0.0, flock, sensed)

    assert_allclose(rates.positions, flock.velocities + array([1.0, -2.0, 0.5]))
    assert_allclose(rates.forces, 0.0)
    assert_allclose(rates.commands, vmap(slots)(sensed))
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
            flock,
            Inbox.from_broadcast(flock, built.radio.range_m, flock),
            still_air(),
            0.0,
            2
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
            flock,
            Inbox.from_broadcast(flock, built.radio.range_m, flock),
            still_air(),
            0.0,
            2
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


@mark.parametrize("rate", [0.0, 1.0, 2.0])
def test_the_heading_turns_a_drone_at_the_rate_its_coupling_sets(rate: float):
    """
    Asserts that the composite hands its heading the rate the coupling sets,
    so a drone whose neighbor flies one meter per second across it turns at
    that rate times the difference under the heading holding the speed.
    """
    view = viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]]))

    assert_allclose(
        composite(coupling=Constant(coupling_per_s=rate), heading=Overdamped())(view),
        [0.0, rate, 0.0],
        atol = 1e-6
    )


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


def test_sensing_holds_each_neighbor_s_heard_state_by_rank():
    """
    Asserts that an observation ranks the neighbors by the distance the
    drone heard and holds each one's offset, distance, velocity and delayed
    velocity as the drone heard them, beside the drone's own position and
    command and the air at its position.
    """
    positions  = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    velocities = array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    flock      = flock_of(positions, velocities, 5 * velocities)
    sent       = flock_of(positions - array([0.5, 0.0, 0.0]), 3 * velocities)

    sensed = Observation.sensed(
        flock,
        Inbox.from_broadcast(flock_of(positions, 2 * velocities), inf, sent),
        still_air(5.0),
        0.0,
        2
    )

    assert_allclose(sensed.offsets[0], [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [0.5, 2.5])
    assert_allclose(sensed.neighbors[0], [[6.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    assert_allclose(sensed.delayed[0], [[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert_array_equal(sensed.position, positions)
    assert_array_equal(sensed.velocity, flock.commands)
    assert_allclose(sensed.air["excess"], 5.0)


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
        flock,
        Inbox.from_broadcast(flock, inf, sent),
        still_air(),
        0.0,
        2
    )

    assert_allclose(sensed.offsets[0], [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [2.0, 4.0])


@mark.parametrize("slot", ["roost", "speed"])
def test_the_composite_adds_each_slot_holding_the_flock_to_its_sum(slot: str):
    """
    Asserts that the composite's command rate carries the roost's turn and
    pull and the speed control beside every other slot, for a drone slow,
    high, and past the rim of the roost.
    """
    view = viewed(
        [[20.0, 0.0, 0.0]] * 7,
        (8.0, 0.0, 0.0),
        position = array([-10.0, 1700.0, 130.0])
    )
    settings = Settings()

    assert_allclose(
        composite()(view) - composite(**{slot: Centroid(stiffness_per_s2=0.0)})(view),
        getattr(settings, slot).scaled(settings.plan.similarity)(view),
        atol = 1e-5
    )


def test_sensing_reads_a_neighbor_out_of_range_as_one_on_the_drone():
    """
    Asserts that a rank no sender heard within the range fills reads as a
    neighbor at the drone's own position flying its command, so the drone
    alone in range adds nothing to a sum over its neighbors.
    """
    positions = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    flock     = flock_of(positions, ones((3, 3)), 2 * ones((3, 3)))

    sensed = Observation.sensed(
        flock,
        Inbox.from_broadcast(flock, 10.0, flock),
        still_air(),
        0.0,
        2
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
        flock,
        Inbox.from_broadcast(flock, 10.0, flock),
        still_air(),
        0.0,
        1
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
        inbox, sent = radio.heard(inbox, sent, flocks[step], key(0), step, 1, 0.1)
        heard[step] = inbox

    assert sent.positions.shape == (5, 3, 3)
    assert_allclose(heard[2].positions, 0.0)
    assert_allclose(heard[4].positions, 1.0)
    assert_allclose(heard[5].positions, 1.0)
    assert_allclose(heard[6].positions, 3.0)
    assert_allclose(heard[6].velocities, 30.0)
    assert_allclose(heard[6].delayed, 20.0)
    assert_allclose(heard[7].positions, 3.0)


def test_the_radio_draws_each_packet_s_loss_apart_at_each_step():
    """
    Asserts that each step draws its own losses, so a packet lost at two
    steps in a row falls at the square of the loss rather than at the loss.
    """
    radio       = Radio(delay_s=0.0, loss=0.4, rate_hz=10.0)
    inbox, sent = radio.opened(flock_of(zeros((200, 3)), zeros((200, 3))), 0, 0.1)
    moved       = flock_of(ones((200, 3)), ones((200, 3)))

    for step in (1, 2):
        inbox, sent = radio.heard(inbox, sent, moved, key(0), step, 0, 0.1)

    assert_allclose(float((inbox.positions[..., 0] == 0.0).mean()), 0.16, atol=0.01)


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
        inbox, sent = radio.heard(inbox, sent, flock, key(0), step, 0, 0.1)
        heard.append(bool(inbox.heard[0, 1]))

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

    assert_array_equal(inbox.heard[0], [True, True, False])
    assert_allclose(
        Observation.sensed(flock, inbox, still_air(), 0.0, 2).offsets[0],
        [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )


def test_the_radio_loses_its_share_of_the_packets_between_two_drones():
    """
    Asserts that each drone hears each other drone's packet with the chance
    the loss leaves, each pair drawn apart, and that a packet lost leaves
    what the drone heard before.
    """
    radio       = Radio(delay_s=0.0, loss=0.4, rate_hz=10.0)
    inbox, sent = radio.opened(flock_of(zeros((200, 3)), zeros((200, 3))), 0, 0.1)

    inbox, _ = radio.heard(
        inbox,
        sent,
        flock_of(ones((200, 3)), ones((200, 3))),
        key(0),
        1,
        0,
        0.1
    )

    assert set(inbox.positions.ravel().tolist()) == {0.0, 1.0}
    assert_allclose(float((inbox.positions[..., 0] == 0.0).mean()), 0.4, atol=0.01)


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

    inbox, sent = radio.heard(inbox, sent, far, key(0), 1, 0, 0.1)
    assert_allclose(inbox.positions[0, 1], [5.0, 0.0, 0.0])

    inbox, sent = radio.heard(inbox, sent, far, key(0), 2, 0, 0.1)
    assert_allclose(inbox.positions[0, 1], [5.0, 0.0, 0.0])
    assert bool(inbox.heard[0, 1])
