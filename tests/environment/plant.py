"""
Pins `Plant`, meaning the lag from a drone's command to its velocity and the
speed and acceleration bounds holding each part of it.
"""

from jax.numpy     import array
from numpy.testing import assert_allclose
from pytest        import mark, param

from thermur.environment.plant import Plant


@mark.parametrize(
    ("command", "velocity", "expected"),
    [
        param((12.0, 0.0, 0.5), (10.0, 0.0, 0.0), (1.0, 0.0, 0.25), id="lag"),
        param((30.0, 40.0, 0.0), (6.0, 8.0, 0.0), (1.5, 2.0, 0.0), id="level-speed"),
        param((0.0, 0.0, 9.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.5), id="climb"),
        param((0.0, 0.0, -9.0), (0.0, 0.0, 0.0), (0.0, 0.0, -0.75), id="descent"),
        param(
            (0.0, 0.0, 0.0),
            (-12.0, -16.0, 0.0),
            (3.0, 4.0, 0.0),
            id = "level-acceleration"
        ),
        param((0.0, 0.0, 3.0), (0.0, 0.0, -1.5), (0.0, 0.0, 2.0), id="climbing"),
        param((0.0, 0.0, -1.5), (0.0, 0.0, 3.0), (0.0, 0.0, -2.0), id="descending")
    ]
)
def test_the_plant_relaxes_a_held_command_under_held_accelerations(
    command  : tuple[float, float, float],
    expected : tuple[float, float, float],
    velocity : tuple[float, float, float]
):
    """
    Asserts that the plant carries a velocity toward its command over the
    lag, the command's horizontal part held to the level speed with its
    direction kept and its vertical part between the descent and the climb,
    and the acceleration's horizontal part held to the level acceleration
    and its vertical part between the descending and climbing ones.
    """
    plant = Plant(
        climb_m_s       = 3.0,
        climbing_m_s2   = 2.0,
        descending_m_s2 = 2.0,
        descent_m_s     = 1.5,
        lag_s           = 2.0,
        level_m_s       = 15.0,
        level_m_s2      = 5.0
    )

    assert_allclose(
        plant.tracked(array(command), array(velocity)),
        expected,
        atol = 1e-6
    )


def test_the_plant_reaches_every_drone_of_a_flock_at_once():
    """
    Asserts that the plant reads a flock's commands and velocities on
    any leading axes, and that a drone already at its command holds its
    velocity.
    """
    commands = array([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2)

    assert_allclose(Plant().tracked(commands, commands), 0.0)
    assert Plant().tracked(commands, commands).shape == (2, 2, 3)
