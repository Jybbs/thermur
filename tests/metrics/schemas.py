"""
Pins `Plan`, meaning the bounds holding a flight's statistics inside it, the
counts it derives from its length and its step, and the launch it draws.
"""

from common.drones import short
from jax.numpy     import array, cos, deg2rad, sin
from jax.random    import key
from numpy.testing import assert_allclose
from pydantic      import ValidationError
from pytest        import mark, param, raises

from thermur.metrics.schemas import Plan


@mark.parametrize(
    ("changes", "match"),
    [
        param({"drones": 7}, "too few drones for 7 neighbors", id="outnumbered"),
        param(
            {"settled_s": 1.6},
            "`settled_s \\+ turning_window_s` of 2.1 s reaches past",
            id = "settled"
        ),
        param(
            {"turn_at_s": 2.0, "turn_deg": 30.0},
            "`turn_at_s` of 2 s reaches past the last frame at 2 s",
            id = "turn"
        ),
        param(
            {"turn_at_s": 0.05, "turn_deg": 30.0},
            "`turn_at_s` of 0.05 s starts before the first frame at 0.1 s",
            id = "early-turn"
        )
    ]
)
def test_a_plan_whose_statistics_read_past_its_flight_is_refused(
    changes : dict[str, float],
    match   : str
):
    """
    Asserts that a plan is refused where its flock holds no more drones than
    a drone has neighbors, or where the settled window or the first drone's
    turn reaches past the last kept frame, naming the value at fault.
    """
    with raises(ValidationError, match=match):
        short(**changes)


def test_a_changed_plan_is_validated_as_a_new_one_is():
    """
    Asserts that `changed` carries each change it is given and refuses one
    that a new plan refuses, where `model_copy` keeps it unchecked.
    """
    plan = short()

    assert plan.changed(drones=20).drones == 20
    with raises(ValidationError, match="too few drones"):
        plan.changed(drones=3)


def test_a_launch_flies_the_course_at_the_cruise_in_a_cloud_about_its_center():
    """
    Asserts that a launch holds every drone at the cruise speed and no
    force, scatters no heading at a zero `scatter_deg`, centers its cloud
    near `center_m`, and draws the same cloud from the same key.
    """
    plan   = short(drones=400, scatter_deg=0.0)
    flock  = plan.launched(key(3))
    spread = flock.positions.std(0)

    assert_allclose((flock.velocities**2).sum(-1), 100.0, rtol=1e-5)
    assert_allclose(flock.velocities, array([[-10.0, 0.0, 0.0]] * 400), atol=1e-5)
    assert_allclose(flock.forces, 0.0)
    assert_allclose(flock.positions.mean(0), plan.center_m, atol=10.0)
    assert spread[0] > spread[1] > spread[2]
    assert_allclose(plan.launched(key(3)).positions, flock.positions)


def test_a_plan_counts_its_steps_and_frames_from_its_length():
    """
    Asserts that a plan counts its integrator's steps and its kept frames
    from its length, its step, and the steps between two frames, the time
    of each frame, a span of seconds in frames and its turning window in at
    least one, and reads its turn in radians.
    """
    plan = short(turn_deg=90.0, turning_deg_s=180.0)

    assert (plan.steps, plan.frames) == (100, 20)
    assert plan.frame_s == 0.1
    assert (plan.framed(0.26), plan.window) == (3, 5)
    assert_allclose(plan.times[[0, -1]], (0.1, 2.0))
    assert short(turning_window_s=0.04).window == 1
    assert_allclose((plan.turn_rad, plan.turning_rad_s), (1.5707963, 3.1415927))


def test_a_plan_s_launch_ellipsoid_holds_its_spacing_cubed_per_drone():
    """
    Asserts that the cloud's standard deviations, as an ellipsoid, hold
    `spacing_m` cubed for each drone, in the proportions `axes` sets.
    """
    spread = Plan(drones=4000, spacing_m=10.0).launched(key(0)).positions.std(0)

    assert_allclose(4 / 3 * 3.14159265 * spread.prod(), 4000 * 10.0**3, rtol=0.05)
    assert_allclose(spread / spread[2], (5.6, 2.8, 1.0), rtol=0.05)


def test_a_turn_late_in_a_plan_with_no_turn_is_kept():
    """
    Asserts that the turn's start bounds no plan whose first drone does not
    turn, so a plan with no turn reaches past its flight's end unrefused.
    """
    assert short(turn_at_s=5.0).turn_at_s == 5.0


def test_the_turned_velocity_holds_the_cruise_along_the_course_plus_the_turn():
    """
    Asserts that the first drone's velocity through its turn points the
    turn's angle counterclockwise from the course, at the cruise speed.
    """
    angle = deg2rad(180.0 + 30.0)

    assert_allclose(
        short(turn_deg=30.0).turned,
        10.0 * array([cos(angle), sin(angle), 0.0]),
        atol = 1e-6
    )
