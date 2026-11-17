"""
Holds the components filling each slot of the controller, each borrowed from
another physical system and cited on the page that explains the choice among
them, meaning the headings, the couplings, the noises, the cohesions, the
one contact, the roost, and the speeds.
"""

from equinox import tree_at
from jax     import Array, grad, hessian
from jax.numpy  import (
    array,
    asarray,
    concat,
    cos,
    cross,
    deg2rad,
    exp,
    expm1,
    finfo,
    full,
    inf,
    linalg,
    log1p,
    maximum,
    minimum,
    select,
    sqrt,
    square,
    where,
    zeros_like
)
from pydantic import Field
from typing   import Annotated, ClassVar, Self

from thermur.controller.schemas  import (
    Component,
    Heading,
    Noise,
    Observation,
    Speed,
    along,
    torque,
    unit
)
from thermur.environment.plant   import Plant
from thermur.environment.schemas import NonNegative, Positive, documented


@documented
class Anisotropic(Component):
    """
    The coupling strengthening a neighbor flying beside a drone over one
    ahead of or behind it, carrying over to alignment the anisotropy of the
    attraction measured between jackdaws.
    """

    across: NonNegative = 1.0
    """
    The added share of `coupling_per_s` for a neighbor directly abeam.
    """

    coupling_per_s: NonNegative = 0.3
    """
    The rate at which a drone aligns with a neighbor directly ahead, in
    reciprocal seconds.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Weights each neighbor by the squared sine of its bearing off the
        drone's velocity.
        """
        return self.coupling_per_s * (1 + self.across * (1 - square(observation.ahead)))


@documented
class Centered(Component):
    """
    The cohesion of a starling flock model, pulling a drone toward its
    topological neighbors outside its rear blind angle and its hard sphere,
    weighted by the drone's centrality over every sender its radio holds,
    which falls near zero for a drone deep inside the flock.
    """

    blind_deg: Annotated[float, Field(ge=0, lt=360)] = 90.0
    """
    The angle behind the drone within which a neighbor is left out, in
    degrees, by default the model's 45 degrees to either side.
    """

    sphere_m: Positive = 0.2
    """
    The radius of the hard sphere within which a neighbor pulls nothing,
    in meters, at a starling's scale, by default half a starling's
    wingspan.
    """

    weight_m_s2: NonNegative = 12.5
    """
    The pull of a drone at a centrality of one, in meters per second
    squared, at a starling's scale, by default the model's cohesion weight
    of 1 newton over its bird's mass of 80 grams.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Multiplies the drone's centrality, the weight, and the sum of the
        bearings of the neighbors it sees past the hard sphere over the
        count of the heard neighbors outside the blind angle, the hard
        sphere's among them.
        """
        counted = (observation.distances > 0) & (
            observation.ahead > -cos(deg2rad(self.blind_deg / 2))
        )
        seen = counted & (observation.distances > self.sphere_m)
        return (
            observation.centrality
            * self.weight_m_s2
            * where(seen[:, None], observation.bearings, 0.0).sum(0)
            / maximum(counted.sum(), 1)
        )

    def scaled(self, factor: float) -> Self:
        """
        Multiplies the hard sphere by `factor` and divides the pull by it,
        as a flock magnified by `factor` at the same speed takes.
        """
        return tree_at(
            lambda centered: (centered.sphere_m, centered.weight_m_s2),
            self,
            (self.sphere_m * factor, self.weight_m_s2 / factor)
        )


@documented
class Centroid(Component):
    """
    The cohesion pulling a drone toward its neighbors' centroid by a spring
    of zero rest length.
    """

    stiffness_per_s2: NonNegative = 0.1
    """
    The acceleration per meter of offset from the centroid, in reciprocal
    square seconds.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Multiplies the mean offset of the neighbors by the stiffness.
        """
        return self.stiffness_per_s2 * observation.offsets.mean(0)


@documented
class Consensus(Heading):
    """
    The spinless second-order velocity consensus, its speed left free,
    which relaxes a drone's velocity toward its neighbors' at the rate the
    coupling sets and toward the mean of those velocities at `centering`
    times the coupling's mean rate.

    The time derivative of the source's position equation gives the second
    term, whose mean over the neighbors stands in for the source's mean over
    the whole flock.
    """

    centering: NonNegative = 2e-4
    """
    The rate at which a drone's velocity relaxes toward its neighbors' mean,
    as a share of the coupling's mean rate per neighbor, by default the
    source's aggregation strength of 0.01 over its alignment strength of 50
    per neighbor, so it carries to any scale the coupling does.
    """

    steering: ClassVar[bool] = False
    """
    Lets the pull change the drone's speed beside the speed slot.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the weighted differences in velocity with each neighbor beside
        the relaxation toward their mean.
        """
        differences = observation.neighbors - observation.velocity
        return observation.weights @ differences + (
            self.centering * observation.weights.mean() * differences.mean(0)
        )


@documented
class Constant(Component):
    """
    The coupling holding one rate over every topological neighbor.
    """

    coupling_per_s: NonNegative = 0.904
    """
    The rate at which a drone aligns with each neighbor, in reciprocal
    seconds, at a starling's scale, by default the relaxation rate inferred
    from the dynamics of starling flocks spread over 7 neighbors,
    `1 / (7 * 0.158)`, where 0.158 seconds is the geometric mean of the
    flocks' relaxation times.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Gives each neighbor the same weight.
        """
        return full(observation.distances.shape, self.coupling_per_s)

    def scaled(self, factor: float) -> Self:
        """
        Divides the rate by `factor`, since a flock magnified by `factor` at
        the same speed takes `factor` times as long over each length.
        """
        return tree_at(
            lambda coupling: coupling.coupling_per_s,
            self,
            self.coupling_per_s / factor
        )


@documented
class Cruise(Speed):
    """
    The speed control of a starling flock model, relaxing a drone's speed
    along its heading toward the cruise speed, so a drone left behind in a
    turn catches up and one pushed ahead falls back.
    """

    cruise_m_s: Positive = 10.0
    """
    The speed the drone relaxes toward, in meters per second, by default the
    model's cruise speed of 10 meters per second, which the plan launches
    the flock at.
    """

    relaxation_s: Positive = 1.0
    """
    The time over which the drone's speed relaxes toward `cruise_m_s`, in
    seconds, at a starling's scale, by default the model's speed control of
    1 second.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Divides the drone's shortfall from `cruise_m_s` by `relaxation_s`,
        directed along its velocity.
        """
        return (
            (self.cruise_m_s - linalg.norm(observation.velocity))
            * unit(observation.velocity)
            / self.relaxation_s
        )

    def scaled(self, factor: float) -> Self:
        """
        Multiplies the relaxation time by `factor`, since a flock magnified
        by `factor` at the same speed takes `factor` times as long over
        each length.
        """
        return tree_at(
            lambda cruise: cruise.relaxation_s,
            self,
            self.relaxation_s * factor
        )

    def stiffness(self, cruise: float, speed: float) -> Array:
        """
        Gives the rate at which the relaxation restores a speed displaced
        from `cruise_m_s`, in reciprocal seconds, the same at every speed.
        """
        return asarray(1 / self.relaxation_s)


@documented
class Delayed(Heading):
    """
    The alignment on neighbor states `lag_s` old, from time-delayed
    dynamics.
    """

    lag_s: Positive = 0.5
    """
    The age of the neighbor states the slot reads, in seconds.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the weighted differences between each neighbor's delayed
        velocity and the drone's own.
        """
        return observation.weights @ (observation.delayed - observation.velocity)


@documented
class Elastic(Component):
    """
    The cohesion joining a drone to each neighbor by a linear spring of
    nonzero rest length, from active elastic matter.
    """

    rest_m: Positive = 18.0
    """
    The spring's rest length, in meters.
    """

    stiffness_per_s2: NonNegative = 0.024
    """
    The acceleration per meter of stretch, in reciprocal square seconds.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums each neighbor's pull along its bearing, a push once it sits
        inside the rest length.
        """
        stretch = self.stiffness_per_s2 * (observation.distances - self.rest_m)
        return stretch @ observation.bearings


@documented
class Fitted(Component):
    """
    The cohesion on the form fitted to pairs of rummy-nose tetras,
    an attraction rising with distance from `rest_m` and falling past
    `range_m`, scaled by the neighbor's bearing off the drone's velocity.
    """

    bearing: NonNegative = 0.5
    """
    The share by which a neighbor ahead pulls harder and one behind pulls
    less.
    """

    range_m: Positive = 40.0
    """
    The distance past which the attraction falls, in meters.
    """

    rest_m: Positive = 18.0
    """
    The distance at which the attraction changes sign, in meters.
    """

    stiffness_per_s2: NonNegative = 0.024
    """
    The acceleration per meter past `rest_m`, in reciprocal square seconds.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums each neighbor's attraction along its bearing.
        """
        distances = observation.distances
        pull      = (
            self.stiffness_per_s2
            * (distances - self.rest_m)
            / (1 + square(distances / self.range_m))
            * (1 + self.bearing * observation.ahead)
        )
        return pull @ observation.bearings


@documented
class HardSphere(Component):
    """
    The separation of a starling flock model, pushing a drone from each
    topological neighbor at full strength inside a hard sphere the width
    of a wingspan and along a halved Gaussian past it, averaged over the
    neighbors.

    The model's equation prints the Gaussian as `exp(-(d - r_h)**2 /
    sigma**2)`, whereas its text calls `sigma` a standard deviation, and the
    slot takes the text's reading.
    """

    sphere_m: Positive = 0.2
    """
    The radius of the hard sphere, in meters, at a starling's scale, by
    default half a starling's wingspan.
    """

    spread_m: Positive = 0.461
    """
    The standard deviation of the Gaussian past the hard sphere, in meters,
    at a starling's scale, by default the one leaving the push at a
    hundredth of its strength at the separation radius of 1.6 meters the
    model sets for its flock of 781 birds, `(1.6 - 0.2) / sqrt(2 * ln 100)`.
    """

    weight_m_s2: NonNegative = 12.5
    """
    The push at full strength, in meters per second squared, at a
    starling's scale, by default the model's separation weight of 1 newton
    over its bird's mass of 80 grams.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Averages over the heard neighbors the push against each bearing,
        `exp(-(d - sphere_m)**2 / (2 * spread_m**2))` past the hard sphere
        and one inside it.
        """
        heard = observation.distances > 0
        past  = maximum(observation.distances - self.sphere_m, 0.0)
        push  = where(heard, exp(-square(past / self.spread_m) / 2), 0.0)
        return -self.weight_m_s2 * (push @ observation.bearings) / maximum(
            heard.sum(),
            1
        )

    def scaled(self, factor: float) -> Self:
        """
        Multiplies the lengths by `factor` and divides the push by it, as a
        flock magnified by `factor` at the same speed takes.
        """
        return tree_at(
            lambda hard: (hard.spread_m, hard.sphere_m, hard.weight_m_s2),
            self,
            (self.spread_m * factor, self.sphere_m * factor, self.weight_m_s2 / factor)
        )


@documented
class Home(Component):
    """
    The roost term of a starling flock model, pulling a drone toward the
    roost's height in proportion to its distance from it, which keeps the
    flock flat, beside the turn fish take from a wall, turning the drone
    across its heading away from the rim of a disk around the roost at a
    rate rising as its time to reach that rim falls.

    The pull also opposes the drone's climb over the ground at twice the
    square root of the stiffness, which damps the height critically.
    """

    bound_m_s2: Positive = Plant().level_m_s2
    """
    The largest turning acceleration, in meters per second squared, by
    default the plant's bound on a horizontal acceleration.
    """

    center_m: tuple[float, float] = (1000.0, 1000.0)
    """
    The disk's center east and north, in meters, by default the launch's.
    """

    height_m: float = 100.0
    """
    The height the drone is pulled toward, in meters above the ground, by
    default the launch's.
    """

    radius_m: Positive = 1000.0
    """
    The disk's radius, in meters.
    """

    stiffness_per_s2: NonNegative = 2.5
    """
    The acceleration toward the height per meter away from it, in
    reciprocal square seconds, at a starling's scale, by default the
    starling model's vertical weight of 0.2 newtons per meter over its
    bird's mass of 80 grams, read per meter since the model multiplies it by
    the height away, although its table of parameters lists it in newtons.
    """

    turn: NonNegative = 0.94
    """
    The turn rate times the time to reach the rim, in radians, by default
    the strength fitted to fish turning from the wall of their tank.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Pulls the drone along the vertical by `vertical`, and turns it
        away from the rim it would reach on its ground velocity's course
        by `turn` times its ground speed over the time to reach it, at most
        `bound_m_s2`, and by `bound_m_s2` outside the disk.
        """
        ground  = (observation.velocity + observation.wind)[:2]
        outward = observation.position[:2] - asarray(self.center_m)
        reach   = self.rim_s(ground, outward)
        heading = unit(ground)
        rate    = where(
            outward @ outward < self.radius_m**2,
            self.turn * linalg.norm(ground) / maximum(reach, finfo(reach).tiny),
            inf
        )
        lateral = -minimum(rate, self.bound_m_s2) * unit(
            outward - outward @ heading * heading
        )
        return concat([lateral, asarray([self.vertical(observation)])])

    def rim_s(self, ground: Array, outward: Array) -> Array:
        """
        Computes the time a drone at `outward` from the disk's center takes
        to reach its rim at the ground velocity `ground`, solving for where
        that course meets the circle of `radius_m`.
        """
        along   = outward @ ground
        squared = ground @ ground
        return (
            sqrt(
                maximum(
                    square(along) - squared * (outward @ outward - self.radius_m**2),
                    0.0
                )
            )
            - along
        ) / maximum(squared, finfo(squared).tiny)

    def scaled(self, factor: float) -> Self:
        """
        Divides the stiffness by the square of `factor`, as a flock
        magnified by `factor` at the same speed takes, and leaves the disk
        and the turn at a drone flock's own values.
        """
        return tree_at(
            lambda home: home.stiffness_per_s2,
            self,
            self.stiffness_per_s2 / factor**2
        )

    def vertical(self, observation: Observation) -> Array:
        """
        Pulls the drone toward `height_m` by `stiffness_per_s2` times its
        height away and against its climb over the ground by twice the
        stiffness's square root.
        """
        return (
            -self.stiffness_per_s2 * (observation.position[2] - self.height_m)
            - 2 * sqrt(
                self.stiffness_per_s2
            ) * (observation.velocity + observation.wind)[2]
        )


@documented
class Intrinsic(Noise):
    """
    The noise of finite numbers, which a drone copying one neighbor at a
    time takes on beside a spontaneous part, its diffusion approximation
    growing with the disagreement among the neighbors.
    """

    copying_per_s: NonNegative = 1.0
    """
    The rate at which a drone copies a neighbor, in reciprocal seconds.
    """

    spontaneous: NonNegative = 3.7
    """
    The amplitude where every neighbor agrees, in meters per second per root
    second.
    """

    def amplitudes(self, observation: Observation) -> tuple[Array, Array]:
        """
        Adds the variance per axis of a copy at `copying_per_s` to the
        spontaneous variance, leaving the force undriven.
        """
        disagreement = square(observation.neighbors - observation.velocity).sum(-1)
        command      = sqrt(
            square(self.spontaneous) + self.copying_per_s * disagreement.mean() / 3
        )
        return command, zeros_like(command)


@documented
class Marginal(Speed):
    """
    The speed channel of marginal confinement, in which a drone imitates its
    neighbors' speeds at the rates the coupling sets and is held about the
    cruise speed by a potential rising as the fourth power of the difference
    of the squared speeds, flat at the cruise speed itself.
    """

    amplitude: NonNegative = 1.0
    """
    The amplitude of the random increments of the speed, in meters per second
    per root second, at a drone flock's scale.
    """

    confinement_per_s: Positive = 0.2
    """
    The strength of the potential, in reciprocal seconds, at a drone flock's
    scale.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the weighted differences between each neighbor's speed and the
        drone's own along its velocity, beside the force of the potential.
        """
        velocity  = observation.velocity
        imitation = observation.weights @ (
            linalg.norm(observation.neighbors, axis=-1) - linalg.norm(velocity)
        )
        return imitation * unit(velocity) - grad(self.potential, argnums=1)(
            observation.cruise,
            velocity
        )

    def kicked(self, increment: Array, observation: Observation) -> Array:
        """
        Scales the part of a unit Brownian increment along the drone's
        command by `amplitude`.
        """
        return self.amplitude * along(increment, observation.velocity)

    def potential(self, cruise: Array, velocity: Array) -> Array:
        """
        Computes the confining potential at `velocity` about the speed
        `cruise`, in square meters per cubic second.
        """
        return (
            self.confinement_per_s / cruise**6 * (velocity @ velocity - cruise**2) ** 4
        )

    def stiffness(self, cruise: float, speed: float) -> Array:
        """
        Computes the curvature of the potential along the velocity at
        `speed`, in reciprocal seconds, the rate at which the force restores
        a speed displaced from it.
        """
        return hessian(self.potential, argnums=1)(
            cruise,
            array([speed, 0.0, 0.0])
        )[0, 0]


@documented
class Metric(Component):
    """
    The coupling holding `Constant`'s rate until the air at the drone runs
    `onset_k` hotter than the ambient, where it weights only the neighbors
    within `range_m`, the rule measured on jackdaws mobbing a predator.
    """

    coupling_per_s: NonNegative = 0.6
    """
    The rate at which a drone aligns with each neighbor it weights, in
    reciprocal seconds.
    """

    onset_k: float = 5.0
    """
    The excess air temperature past which the rule turns metric, in kelvin.
    """

    range_m: Positive = 25.0
    """
    The distance within which a neighbor counts while the rule is metric, in
    meters.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Weights each neighbor by the rate, dropping those past `range_m`
        once the excess reaches the onset.
        """
        return self.coupling_per_s * where(
            (observation.air["excess"] >= self.onset_k)
            & (observation.distances > self.range_m),
            0.0,
            1.0
        )


@documented
class Overdamped(Heading):
    """
    The overdamped limit of the inertial spin model at a constant speed, in
    which a drone's heading relaxes toward its neighbors' with no spin.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the weighted differences between each neighbor's velocity and
        the drone's own.
        """
        return observation.weights @ (observation.neighbors - observation.velocity)


@documented
class InertialSpin(Overdamped):
    """
    The inertial spin model, in which the pull of the neighbors drives a
    spin under an inertia and a friction and the spin turns the command, so
    a turn travels as a wave rather than spreading by diffusion.

    The other slots' forces turn the command directly, as a force on the
    command rather than on the spin, and the model reduces to `Overdamped`
    at a `friction` of one as `inertia_s` falls to zero.
    """

    friction: Positive = 0.3
    """
    The friction on the spin, which relaxes it at `friction / inertia_s`.
    """

    inertia_s: Positive = 0.0112
    """
    The inertia of the spin, in seconds, at a starling's scale, by default
    the one carrying a turn across the launch at 30 meters per second, the
    middle of the 20 to 40 measured on turning starling flocks, `7 * 0.904 *
    1.264**2 / 30**2`, where 1.264 meters is the launch's root-mean-square
    distance to its seven neighbors at a starling's scale.
    """

    def kicked(self, increment: Array, observation: Observation) -> tuple[Array, Array]:
        """
        Passes the torque of a Brownian increment of the command across
        it to the spin, scaled by the square root of `friction` over
        `inertia_s`, which leaves the overdamped heading's noise at a
        `friction` of one as `inertia_s` falls to zero.

        Args:
            increment   : The increment of the command, in meters per
                          second.
            observation : What the drone senses.

        Returns:
            The increment of the drone's command, in meters per second, and
            of its spin, in radians per second.
        """
        return (
            zeros_like(increment),
            sqrt(self.friction) * torque(
                increment,
                observation.velocity
            ) / self.inertia_s
        )

    def scaled(self, factor: float) -> Self:
        """
        Multiplies the inertia by `factor`, as a flock magnified by `factor`
        at the same speed takes, leaving the friction, a ratio of rates, as
        it stands.
        """
        return tree_at(
            lambda spin: spin.inertia_s,
            self,
            self.inertia_s * factor
        )

    def steered(self, force: Array, observation: Observation) -> tuple[Array, Array]:
        """
        Turns the command at the drone's spin and at the rate `force` turns
        it, which carries the spin with the command, and relaxes the spin
        toward the torque of the pull.

        Args:
            force       : The other slots' acceleration on the drone, in
                          meters per second squared.
            observation : What the drone senses, its weights set.

        Returns:
            The rate of the drone's command, in meters per second squared,
            and of its spin, in radians per second squared.
        """
        command = observation.velocity
        spin    = observation.spin
        turning = torque(force, command)
        return (
            cross(spin + turning, command),
            (torque(
                self(observation),
                command
            ) - self.friction * spin) / self.inertia_s
            + cross(turning, spin)
        )


@documented
class Nonmutual(InertialSpin):
    """
    The inertial spin under nonmutual torques, in which a neighbor ahead of
    a drone turns it harder than one behind does, so the torque two drones
    exert on each other differs.
    """

    asymmetry: Annotated[float, Field(ge=0, le=1)] = 0.5
    """
    The added share of a neighbor's weight per unit cosine of its bearing off
    the drone's velocity, which takes the weight of a neighbor straight behind
    to zero at one.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the differences in velocity with each neighbor, weighted by the
        coupling and by the neighbor's bearing.
        """
        return observation.weights * (1 + self.asymmetry * observation.ahead) @ (
            observation.neighbors - observation.velocity
        )


@documented
class Persistent(Noise):
    """
    The random force of an inertial active Ornstein-Uhlenbeck particle,
    relaxing with a persistence time and entering the acceleration as a
    continuous term.
    """

    persistence_s: Positive = 1.0
    """
    The time over which the force decorrelates, in seconds.
    """

    strength_m_s2: NonNegative = 3.8
    """
    The root-mean-square force per axis, in meters per second squared.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Adds the force the drone holds to its acceleration.
        """
        return observation.force

    def amplitudes(self, observation: Observation) -> tuple[Array, Array]:
        """
        Drives the force alone, at `strength_m_s2 * sqrt(2 /
        persistence_s)`.
        """
        force = self.strength_m_s2 * sqrt(2 / self.persistence_s)
        return zeros_like(force), force

    def relaxation(self, observation: Observation) -> Array:
        """
        Decays the force over `persistence_s`.
        """
        return -observation.force / self.persistence_s


@documented
class Quartic(InertialSpin):
    """
    The inertial spin under an alignment adding a quartic term to the
    quadratic one, so a neighbor's pull grows with the square of its
    difference in velocity, and a large turn crosses the flock as a front
    whose speed rises with its amplitude while small fluctuations stay
    damped.
    """

    friction: Positive = 0.6
    """
    The friction on the spin, which relaxes it at `friction / inertia_s`.
    """

    quartic: NonNegative = 10.0
    """
    The strength of the quartic term over the quadratic one, at a difference
    in velocity equal to the cruise speed.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the differences in velocity with each neighbor, each weighted
        by the coupling and grown by the quartic term.
        """
        differences = observation.neighbors - observation.velocity
        growth = self.quartic * square(differences).sum(-1) / square(observation.cruise)
        return observation.weights * (1 + growth) @ differences


@documented
class Vectorial(Noise):
    """
    The noise added to the neighbor sum before it is normalized, whose
    influence falls as local order rises.
    """

    amplitude: NonNegative = 4.1
    """
    The amplitude where the neighbors fly as one, in meters per second per
    root second.
    """

    floor: Positive = 0.3
    """
    The smallest length the mean neighbor heading divides by, near the length
    of the mean of a neighbor count of random headings.
    """

    def amplitudes(self, observation: Observation) -> tuple[Array, Array]:
        """
        Divides the amplitude by the length of the mean neighbor heading,
        leaving the force undriven.
        """
        order   = linalg.norm(unit(observation.neighbors).mean(0))
        command = self.amplitude / maximum(order, self.floor)
        return command, zeros_like(command)


@documented
class Weighted(Component):
    """
    The coupling decaying with distance even within a rank, after the
    interactions by physical distance that chimney swifts show.
    """

    coupling_per_s: NonNegative = 1.2
    """
    The rate at which a drone would align with a neighbor at no distance, in
    reciprocal seconds.
    """

    range_m: Positive = 20.0
    """
    The distance over which the rate falls by a factor of e, in meters.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Weights each neighbor by the decay of the rate over its distance.
        """
        return self.coupling_per_s * exp(-observation.distances / self.range_m)


@documented
class White(Noise):
    """
    One amplitude for every drone, entering as a Brownian increment on the
    command.
    """

    amplitude: NonNegative = 2.455
    """
    The amplitude, in meters per second per root second, at a starling's
    scale, by default `10 * sqrt(2 * T)` at 10 meters per second, the
    temperature `T` of 0.0301 being the relaxation rate `1 / 0.158` over
    the product of 12.13 and 17.31, the medians of the interaction range
    and of the coupling over the temperature inferred from starling flocks.
    """

    def amplitudes(self, observation: Observation) -> tuple[Array, Array]:
        """
        Drives every command at one amplitude, leaving the force undriven.
        """
        command = asarray(self.amplitude)
        return command, zeros_like(command)

    def scaled(self, factor: float) -> Self:
        """
        Divides the amplitude by the square root of `factor`, holding the
        flock's alignment against its noise as the coupling's rate falls
        by `factor`.
        """
        return tree_at(
            lambda white: white.amplitude,
            self,
            self.amplitude / sqrt(factor)
        )


@documented
class Zig(White):
    """
    The white noise on the command beside the zig of a starling flock
    model's waves of agitation, a push to one side across the heading and
    back that a drone starts once the roost turns it as hard as the zig
    would, or copies from the neighbors it hears zigging, each zigging
    neighbor setting it off within one zig with the chance `branching` over
    the neighbor count, the branching ratio of an excitable network.

    The noise state holds the countdown to the end of the refractory time
    after the zig in seconds, the side it pushes toward, one to the left of
    the heading and minus one to the right, and zero.
    """

    accel_m_s2: NonNegative = 12.5
    """
    The push across the heading, in meters per second squared, at a
    starling's scale, by default the model's zig force of 1 newton over its
    bird's mass of 80 grams.
    """

    back_s: Positive = 0.30
    """
    The time the drone pushes back toward its first heading, in seconds,
    at a starling's scale.
    """

    branching: NonNegative = 1.0
    """
    The mean number of drones each zig sets off among the neighbors hearing
    it, by default the critical branching ratio of an excitable network.
    """

    cue_s: NonNegative = 0.05
    """
    The time a drone takes to recognize a neighbor's zig, in seconds, at a
    starling's scale.
    """

    reaction_s: NonNegative = 0.076
    """
    The time between a cue and the start of the zig, in seconds, at a
    starling's scale, by default the reaction time of a starling.
    """

    refractory_s: NonNegative = 1.0
    """
    The time after a zig during which the drone starts none, in seconds, at
    a starling's scale.
    """

    side_s: Positive = 0.25
    """
    The time the drone pushes to its side, in seconds, at a starling's
    scale.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Pushes the drone across its heading by `accel_m_s2` toward its
        side for `side_s` and away from it for `back_s`, once the countdown
        enters the zig, and adds nothing through the wait and the refractory
        time around it.
        """
        countdown, side, _ = observation.force
        return where(
            self.zigging(countdown),
            side * self.accel_m_s2 * where(
                self.refractory_s + self.back_s < countdown,
                1.0,
                -1.0
            ),
            0.0
        ) * left(observation.velocity)

    @property
    def zig_s(self) -> float:
        """
        Adds the time the drone pushes to its side to the time it pushes
        back, in seconds.
        """
        return self.side_s + self.back_s

    def excited(
        self,
        delay_s     : float,
        draw        : Array,
        observation : Observation,
        step_s      : float,
        turn        : Array
    ) -> Array:
        """
        Starts a resting drone's zig away from the rim once the roost's
        horizontal turn reaches `accel_m_s2`, after `reaction_s`, or else
        toward the side most of its zigging neighbors push where `draw`
        falls below the chance their zigs set it off within the step, after
        `reaction_s` and `cue_s` less the radio's `delay_s`, and otherwise
        counts the countdown down by `step_s`.
        """
        countdown, side, _ = observation.force
        start              = self.refractory_s + self.zig_s
        zigging            = (observation.distances > 0) & self.zigging(
            observation.forces[:, 0]
        )
        hazard  = -log1p(-self.branching / observation.distances.size) / self.zig_s
        resting = countdown <= 0
        walled  = resting & (linalg.norm(turn[:2]) >= self.accel_m_s2)
        copied  = resting & ~walled & (
            draw < -expm1(-hazard * zigging.sum() * step_s)
        )
        return select(
            [walled, copied],
            [
                asarray(
                    [
                        start + self.reaction_s,
                        where(turn @ left(observation.velocity) >= 0, 1.0, -1.0),
                        0.0
                    ]
                ),
                asarray(
                    [
                        start + maximum(self.reaction_s + self.cue_s - delay_s, 0.0),
                        where(observation.forces[:, 1] @ zigging >= 0, 1.0, -1.0),
                        0.0
                    ]
                )
            ],
            asarray([maximum(countdown - step_s, 0.0), side, 0.0])
        )

    def scaled(self, factor: float) -> Self:
        """
        Multiplies the times by `factor`, divides the push by it, and the
        white noise's amplitude by its square root, as a flock magnified by
        `factor` at the same speed takes.
        """
        return tree_at(
            lambda zig: (
                zig.accel_m_s2, zig.back_s, zig.cue_s,
                zig.reaction_s, zig.refractory_s, zig.side_s
            ),
            super().scaled(factor),
            (
                self.accel_m_s2 / factor,
                self.back_s * factor,
                self.cue_s * factor,
                self.reaction_s * factor,
                self.refractory_s * factor,
                self.side_s * factor
            )
        )

    def zigging(self, countdown: Array) -> Array:
        """
        Tells whether a drone holding `countdown` is pushing to its side or
        back, the countdown having passed the wait and not yet reached the
        refractory time.
        """
        return (countdown > self.refractory_s) & (
            countdown <= self.refractory_s + self.zig_s
        )


def left(velocity: Array) -> Array:
    """
    Turns the horizontal part of `velocity` a quarter turn to the left and
    divides it by its length.
    """
    return unit(cross(asarray([0.0, 0.0, 1.0]), velocity))
