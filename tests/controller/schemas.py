"""
Pins the core's records, meaning the properties of a `Flock`, the neighbors
an `Observation` holds, and how a `Composite` sums its slots.
"""

from common.drones import flock_of, still_air, viewed
from dataclasses   import fields, replace
from jax           import vmap
from jax.numpy     import array, eye, ones, zeros
from lineax        import DiagonalLinearOperator
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
    Noise,
    Observation,
    across,
    documented,
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
    Asserts that the operator the Brownian path enters through holds zero on
    each position and each amplitude the noise gives.
    """
    flock  = flock_of(zeros((8, 3)), ones((8, 3)))
    sensed = Observation.sensed(flock, flock.velocities, still_air(), 0.0, 7)

    white      = composite(noise=White(amplitude=2.0)).diffusion(0.0, flock, sensed)
    persistent = composite(
        noise = Persistent(persistence_s=0.5, strength_m_s2=3.0)
    ).diffusion(
        0.0, flock, sensed
    )

    assert isinstance(white, DiagonalLinearOperator)
    assert_allclose(white.diagonal.positions, 0.0)
    assert_allclose(white.diagonal.velocities, 2.0)
    assert_allclose(white.diagonal.forces, 0.0)
    assert_allclose(persistent.diagonal.velocities, 0.0)
    assert_allclose(persistent.diagonal.forces, 6.0)


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
    sensed    = Observation.sensed(flock, flock.velocities, still_air(), 0.0, 7)
    slots     = composite(noise=Persistent(persistence_s=2.0))

    rates = slots.drift(0.0, flock, sensed)

    assert_array_equal(rates.positions, flock.velocities)
    assert_allclose(rates.forces, 0.0)
    assert_allclose(rates.velocities, vmap(slots)(sensed))


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
    position.
    """
    positions  = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    velocities = array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    flock      = flock_of(positions, velocities)

    sensed = Observation.sensed(flock, 2 * velocities, still_air(5.0), 0.0, 2)

    assert_allclose(sensed.offsets[0], [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert_allclose(sensed.distances[0], [1.0, 3.0])
    assert_allclose(sensed.neighbors[0], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert_allclose(sensed.delayed[0], [[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert_allclose(sensed.air["excess"], 5.0)


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
