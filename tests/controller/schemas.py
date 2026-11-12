"""
Pins the core's records, meaning the properties of a `Flock`, what the
`Radio` delivers and loses, the neighbors an `Observation` holds, and how a
`Composite` sums its slots.
"""

from common.drones import flock_of, still_air, viewed
from dataclasses   import replace
from jax           import vmap
from jax.numpy     import array, eye, full, ones, zeros
from jax.random    import key
from lineax        import DiagonalLinearOperator
from math          import inf
from numpy.testing import assert_allclose, assert_array_equal
from pytest        import mark, raises

from thermur.controller.components import (
    Consensus,
    Constant,
    Morse,
    Overdamped,
    Persistent,
    SoftCore,
    White
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
        "cohesion" : SoftCore(diameter_m=30.0),
        "noise"    : White(),
        "contact"  : Morse()
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
        Inbox.from_broadcast(flock, flock),
        still_air(),
        0.0,
        7,
        inf
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
        Inbox.from_broadcast(flock, flock),
        still_air(wind=(1.0, -2.0, 0.5)),
        0.0,
        7,
        inf
    )
    slots = composite(noise=Persistent(persistence_s=2.0))

    rates = slots.drift(0.0, flock, sensed)

    assert_allclose(rates.positions, flock.velocities + array([1.0, -2.0, 0.5]))
    assert_allclose(rates.forces, 0.0)
    assert_allclose(rates.commands, vmap(slots)(sensed))
    assert_allclose(rates.velocities, [[0.0, 2.0, 0.0]] * 8, rtol=1e-6)


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


def test_sensing_holds_each_neighbor_s_heard_state_by_rank():
    """
    Asserts that an observation ranks the neighbors by their true distance
    and holds each one's offset, distance, velocity and delayed velocity as
    the drone heard them, beside the drone's own position and command and
    the air at its position.
    """
    positions  = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    velocities = array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    flock      = flock_of(positions, velocities, 5 * velocities)
    sent       = flock_of(positions - array([0.5, 0.0, 0.0]), 3 * velocities)

    sensed = Observation.sensed(
        flock,
        Inbox.from_broadcast(flock_of(positions, 2 * velocities), sent),
        still_air(5.0),
        0.0,
        2,
        inf
    )

    assert_allclose(sensed.offsets[0], [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [0.5, 2.5])
    assert_allclose(sensed.neighbors[0], [[6.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    assert_allclose(sensed.delayed[0], [[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert_array_equal(sensed.position, positions)
    assert_array_equal(sensed.velocity, flock.commands)
    assert_allclose(sensed.air["excess"], 5.0)


def test_sensing_reads_a_neighbor_out_of_range_as_one_on_the_drone():
    """
    Asserts that a neighbor the query finds none for within the range reads
    as one at the drone's own position flying its command, so the drone
    alone in range adds nothing to a sum over its neighbors.
    """
    positions = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    flock     = flock_of(positions, ones((3, 3)), 2 * ones((3, 3)))

    sensed = Observation.sensed(
        flock,
        Inbox.from_broadcast(flock, flock),
        still_air(),
        0.0,
        2,
        10.0
    )

    assert_allclose(sensed.offsets[0], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert_allclose(sensed.offsets[2], 0.0)
    assert_allclose(sensed.neighbors[0], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert_allclose(sensed.delayed[2], 2.0)


@mark.parametrize("rate", [0.0, 1.0, 2.0])
def test_the_heading_turns_a_drone_at_the_rate_its_coupling_sets(rate: float):
    """
    Asserts that the composite hands its heading the rate the coupling sets,
    so a drone whose neighbor flies one meter per second across it turns at
    that rate times the difference.
    """
    view = viewed([[10.0, 0.0, 0.0]], neighbors=array([[10.0, 1.0, 0.0]]))

    assert_allclose(
        composite(coupling=Constant(coupling_per_s=rate))(view),
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


def test_summing_a_neighbor_on_the_drone_stays_finite():
    """
    Asserts that a neighbor at the drone's exact position takes a bearing of
    zero, so a slot reading it stays finite.
    """
    assert_allclose(viewed([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).bearings[0], 0.0)


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
