"""
Holds the components filling each slot of the controller, each borrowed from
another physical system and cited on the page that explains the choice among
them, meaning the headings, the couplings, the noises, the cohesions, the
one contact, the arena, the roost, and the speed control.
"""

from equinox import tree_at
from jax     import Array
from jax.numpy  import (
    absolute,
    asarray,
    cos,
    deg2rad,
    exp,
    eye,
    full,
    linalg,
    maximum,
    newaxis,
    sign,
    sqrt,
    square,
    where,
    zeros_like
)
from pydantic import Field
from typing   import Annotated, ClassVar, Self

from thermur.controller.schemas  import Component, Heading, Noise, Observation, unit
from thermur.environment.schemas import Grid, NonNegative, Positive, documented


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
class Arena(Component):
    """
    The virtual walls of a drone flocking model, one shill agent on each
    face of a box nearest the drone, moving straight into the box at
    `shill_m_s`, whose velocity the drone relaxes toward wherever its
    velocity differs from the shill's by more than a braking curve allows at
    its distance from that face.
    """

    braking_m_s2: Positive = 4.18
    """
    The acceleration of the braking curve, in meters per second squared,
    the model's optimized wall acceleration for a flock at 8 meters per
    second.
    """

    extent_m: tuple[Positive, Positive, Positive] = Grid().extent_m
    """
    The box's far corner east, north, and above the ground, in meters, its
    near corner at the origin, by default the parametric threat's grid.
    """

    gain_per_s: Positive = 8.12
    """
    The slope of the braking curve's linear part near the wall, in
    reciprocal seconds, the model's optimized wall gain for a flock at 8
    meters per second.
    """

    offset_m: NonNegative = 0.1
    """
    The distance inside each face at which the braking curve stops, in
    meters, the model's optimized wall offset for a flock at 8 meters per
    second.
    """

    relaxation_s: Positive = 1.0
    """
    The time over which the drone's velocity relaxes toward the shill's
    excess, in seconds, the model's velocity relaxation time.
    """

    shill_m_s: Positive = 22.8
    """
    The speed of each shill agent into the box, in meters per second, the
    model's optimized shill speed for a flock at 8 meters per second.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums, over the nearer face on each axis, the velocity difference
        from that face's shill in excess of the braking curve at the drone's
        distance inside the face, directed from the drone's velocity toward
        the shill's, divided by `relaxation_s`.
        """
        middle    = asarray(self.extent_m) / 2
        inward    = middle - observation.position
        inside    = middle - absolute(inward) - self.offset_m
        shills    = self.shill_m_s * eye(3) * sign(inward)[:, newaxis]
        gaps      = shills - observation.velocity
        lengths   = linalg.norm(gaps, axis=-1)
        crossover = self.braking_m_s2 / self.gain_per_s
        allowed   = where(
            inside * self.gain_per_s < crossover,
            maximum(inside, 0.0) * self.gain_per_s,
            sqrt(maximum(2 * self.braking_m_s2 * inside - crossover**2, 0.0))
        )
        excess = maximum(lengths - allowed, 0.0)[:, newaxis] * unit(gaps)
        return excess.sum(0) / self.relaxation_s

    def scaled(self, _: float) -> Self:
        """
        Leaves the walls as they stand, since their values are a drone's own
        at any spacing.
        """
        return self


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

    confined: ClassVar[bool] = False
    """
    Leaves each drone's speed to the sum of the slots.
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

    coupling_per_s: NonNegative = 3.571
    """
    The rate at which a drone aligns with each neighbor, in reciprocal
    seconds, at a starling's scale, by default one step of the projection
    model spread over 7 neighbors, since that model replaces a bird's
    heading in full at every step, its step the 0.04 seconds a starling at
    10 meters per second takes to fly its wingspan of 0.4 meters.
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
class Cruise(Component):
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
class Peripheral(Component):
    """
    The cohesion of a starling flock model, pulling a drone toward its
    topological neighbors outside its rear blind angle and its hard sphere,
    weighted by how near the flock's border the drone sits, so a drone
    deep inside the flock feels almost none and the flock keeps its spacing
    rather than drawing in.

    The drone's centrality is the length of the mean bearing of its heard
    neighbors, near zero inside the flock and above a half at its border.
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
            linalg.norm(observation.bearings.sum(0))
            / maximum((observation.distances > 0).sum(), 1)
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
            lambda peripheral: (peripheral.sphere_m, peripheral.weight_m_s2),
            self,
            (self.sphere_m * factor, self.weight_m_s2 / factor)
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
class Roost(Component):
    """
    The vertical part of a starling flock model's roost term, pulling a
    drone toward the roost's height in proportion to its distance from it,
    which keeps the flock flat.
    """

    height_m: float = 100.0
    """
    The height the drone is pulled toward, in meters above the ground, by
    default the launch's.
    """

    stiffness_per_s2: NonNegative = 2.5
    """
    The acceleration toward the height per meter away from it, in
    reciprocal square seconds, at a starling's scale, by default the
    starling model's vertical weight of 0.2 newtons per meter over its
    bird's mass of 80 grams, read per meter since the model multiplies it by
    the height away, although its table of parameters lists it in newtons.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Pulls the drone along the vertical by `stiffness_per_s2` times its
        height away from `height_m`.
        """
        return asarray([0.0, 0.0, -self.stiffness_per_s2]) * (
            observation.position[2] - self.height_m
        )

    def scaled(self, factor: float) -> Self:
        """
        Divides the stiffness by the square of `factor`, as a flock
        magnified by `factor` at the same speed takes.
        """
        return tree_at(
            lambda roost: roost.stiffness_per_s2,
            self,
            self.stiffness_per_s2 / factor**2
        )


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

    amplitude: NonNegative = 3.15
    """
    The amplitude, in meters per second per root second, at a starling's
    scale, by default the noise that leaves a flock aligning at
    `Constant`'s default rate over 7 neighbors at the product of coupling
    and interaction range of 503 inferred from starling flocks,
    `10 * sqrt(2 * 7 * 3.571 / 503)` at 10 meters per second.
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
