"""
Pins `Plan`, meaning the bounds holding a flight's statistics inside it, the
counts it derives from its length and its step, and the launch it draws.
"""

from common.drones import short
from jax           import vmap
from jax.numpy     import arccos, array, cos, deg2rad, sin, sqrt
from jax.random    import key, split
from numpy         import arange
from numpy.testing import assert_allclose
from pydantic      import ValidationError
from pytest        import mark, param, raises
from scipy.spatial import KDTree

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


def test_a_flock_of_100_launches_at_the_center_s_height_as_drawn():
    """
    Asserts that a flock of 100 launches at the height `center_m` sets with
    every drone of 256 launches above the ground, so neither the raised
    height nor the hold at the ground moves a drone off its draw.
    """
    plan = Plan(drones=100)

    assert plan.height_m == plan.center_m[2]
    assert vmap(plan.launched)(split(key(plan.seed), 256)).positions[..., 2].min() > 0


def test_a_flock_of_the_default_size_launches_above_the_ground():
    """
    Asserts that a flock of the default size launches its center
    `clearance_sd` deviations of its depth above the ground, past the height
    `center_m` sets, and that no drone of 256 launches starts under the
    ground, the hold at the ground setting at least one drone and fewer
    than one in 10,000 on it, each climbing and commanding the velocity it
    launches at.
    """
    plan     = Plan()
    launches = vmap(plan.launched)(split(key(plan.seed), 256))
    heights  = launches.positions[..., 2]

    assert plan.height_m == plan.clearance_sd * plan.depth_m > plan.center_m[2]
    assert_allclose(heights.std(1).mean(), plan.depth_m, rtol=0.05)
    assert heights.min() >= 0
    assert 0 < (heights == 0).sum() < 1e-4 * heights.size
    assert (launches.velocities[..., 2][heights == 0] > 0).all()
    assert (launches.commands == launches.velocities).all()


def test_a_launch_flies_the_course_at_the_cruise_in_a_cloud_about_its_center():
    """
    Asserts that a launch holds every drone at the cruise speed and no
    force, scatters no heading at a zero `scatter_deg`, centers its cloud
    near `center_m` at `height_m`, and draws the same cloud from the same
    key.
    """
    plan   = short(drones=400, scatter_deg=0.0)
    flock  = plan.launched(key(3))
    spread = flock.positions.std(0)

    assert_allclose((flock.velocities**2).sum(-1), 100.0, rtol=1e-5)
    assert_allclose(flock.velocities, array([[-10.0, 0.0, 0.0]] * 400), atol=1e-5)
    assert_allclose(flock.forces, 0.0)
    assert_allclose(
        flock.positions.mean(0),
        (*plan.center_m[:2], plan.height_m),
        atol = 10.0
    )
    assert spread[0] > spread[1] > spread[2]
    assert_allclose(plan.launched(key(3)).positions, flock.positions)


def test_a_launch_magnifies_the_starling_flock_by_its_nearest_neighbor_distance():
    """
    Asserts that the plan's similarity is the launch's nearest-neighbor
    distance over the starling flock's, doubling with the spacing.
    """
    plan = Plan(spacing_m=15.0, starling_m=0.5)

    assert_allclose(plan.similarity, plan.nearest_m / 0.5)
    assert_allclose(
        Plan(spacing_m=30.0, starling_m=0.5).similarity,
        2 * plan.similarity
    )


def test_a_launch_scatters_each_heading_about_the_course_by_scatter_deg():
    """
    Asserts that a launch at a `scatter_deg` of 10 tilts each heading off
    the course by a root-mean-square angle near 10 degrees times the root of
    two, the two axes across the course each drawing that spread.
    """
    velocities = short(drones=400, scatter_deg=10.0).launched(key(3)).velocities

    assert_allclose(
        sqrt((arccos(-velocities[:, 0] / 10.0) ** 2).mean()),
        deg2rad(10.0) * sqrt(2.0),
        rtol = 0.1
    )


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


@mark.parametrize(
    "axes",
    [param((5.6, 2.8, 1.0), id="flat"), param((1.0, 1.0, 1.0), id="round")]
)
def test_a_launch_keeps_the_nearest_neighbor_distance_its_density_sets(
    axes: tuple[float, float, float]
):
    """
    Asserts that a launch of 1000 drones holds its drones a mean distance
    from their nearest neighbors within 3% of `nearest_m`, whatever its
    shape.
    """
    plan  = Plan(axes=axes, drones=1000)
    cloud = plan.launched(key(0)).positions

    assert_allclose(
        KDTree(cloud).query(cloud, 2)[0][:, 1].mean(),
        plan.nearest_m,
        rtol = 0.03
    )


def test_the_shape_frames_sit_a_shape_interval_apart_through_the_settled_flight():
    """
    Asserts that a flight of 100 s reads its shape each second from 21 s,
    the first whole second past `settled_s`, through its last frame.
    """
    plan = Plan(duration_s=100.0)

    assert_allclose(plan.times[plan.shape_frames], arange(21.0, 101.0))


@mark.parametrize("every_s", [0.1, 0.5])
def test_a_shape_interval_within_one_frame_reads_every_settled_frame(every_s: float):
    """
    Asserts that a shape interval no longer than a kept frame reads every
    frame from `settled_s` on.
    """
    plan = Plan(duration_s=30.0, shape_every_s=every_s)

    assert_allclose(plan.times[plan.shape_frames], arange(20.5, 30.5, 0.5))


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
