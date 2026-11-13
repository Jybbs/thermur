"""
Holds the components filling each slot of the controller, each borrowed from
another physical system and cited on the page that explains the choice among
them, meaning the headings, the couplings, the noises, the cohesions, the
one contact, and the one roost.
"""

from equinox import tree_at
from jax     import Array
from jax.numpy  import (
    append,
    asarray,
    cos,
    deg2rad,
    exp,
    full,
    linalg,
    maximum,
    sqrt,
    square,
    where,
    zeros_like
)
from pydantic import Field
from typing   import Annotated, ClassVar, Self

from thermur.controller.schemas  import Component, Heading, Noise, Observation, unit
from thermur.environment.schemas import GRAVITY_M_S2, NonNegative, Positive, documented


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
    which relaxes a drone's velocity toward its neighbors' at the
    rate the coupling sets and toward the mean of those velocities at
    `centering_per_s`.

    The time derivative of the source's position equation gives the second
    term, whose mean over the neighbors stands in for the source's mean over
    the whole flock.
    """

    centering_per_s: NonNegative = 0.1
    """
    The rate at which a drone's velocity relaxes toward its neighbors' mean,
    in reciprocal seconds.
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
        return (
            observation.weights @ differences
            + self.centering_per_s * differences.mean(0)
        )


@documented
class Constant(Component):
    """
    The coupling holding one rate over every topological neighbor.
    """

    coupling_per_s: NonNegative = 2.857
    """
    The rate at which a drone aligns with each neighbor, in reciprocal
    seconds, at a starling's scale, by default the projection model's
    alignment weight of 0.8 per step spread over 7 neighbors, its step the
    time a starling at 10 meters per second takes to fly its wingspan of
    0.4 meters.
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
    """

    sphere_m: Positive = 0.2
    """
    The radius of the hard sphere, in meters, at a starling's scale, by
    default half a starling's wingspan.
    """

    spread_m: Positive = 1.37
    """
    The standard deviation of the Gaussian past the hard sphere, in meters,
    at a starling's scale, by default the model's, which leaves the push at
    about a fiftieth of its strength at the model's separation radius of 4
    meters.
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
        Multiplies the drone's centrality, the weight, and the mean bearing
        of the neighbors it sees past the hard sphere.
        """
        seen = (
            (observation.distances > self.sphere_m)
            & (observation.ahead > -cos(deg2rad(self.blind_deg / 2)))
        )
        return (
            linalg.norm(observation.bearings.sum(0))
            / maximum((observation.distances > 0).sum(), 1)
            * self.weight_m_s2
            * where(seen[:, None], observation.bearings, 0.0).sum(0)
            / maximum(seen.sum(), 1)
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
    The roost term of a starling flock model, turning a drone that has left
    a vertical cylinder about the roost across its heading, the harder the
    further its heading points away from the roost, toward the side that
    points it back, and pulling it toward the roost's height through a
    damped spring.
    """

    center_m: tuple[float, float] = (600.0, 1000.0)
    """
    The roost's east and north, in meters, by default on the parametric
    threat's course from the launch to the domain's western edge.
    """

    height_m: float = 100.0
    """
    The height the drone is pulled toward, in meters above the ground.
    """

    radius_m: Positive = 600.0
    """
    The radius of the cylinder, in meters, by default the widest the
    parametric threat's domain holds about `center_m`.
    """

    stiffness_per_s2: NonNegative = 2.5
    """
    The acceleration toward the height per meter away from it, in
    reciprocal square seconds, at a starling's scale, by default the
    starling model's vertical weight of 0.2 newtons per meter over its
    bird's mass of 80 grams, read per meter since the model multiplies it by
    the height away, although its table of parameters lists it in newtons.
    """

    turning_m_s2: NonNegative = GRAVITY_M_S2
    """
    The acceleration turning a drone headed straight out of the cylinder, in
    meters per second squared, half of it for a drone headed along the
    cylinder's edge, at a starling's scale, by default the acceleration of
    gravity, since the model's roost term rolls a starling past the bank its
    roll-out balances, so the bird turns on the whole of a lift equal to its
    weight.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the horizontal turn across the drone's heading, toward the
        side that points it back at the roost, and the vertical pull toward
        `height_m`, damped at `2 * sqrt(stiffness_per_s2)` on the drone's
        commanded climb.
        """
        away    = observation.position[:2] - asarray(self.center_m)
        outward = unit(away)
        heading = unit(observation.velocity[:2])
        left    = asarray([-heading[1], heading[0]])
        turn    = where(
            linalg.norm(away) > self.radius_m,
            self.turning_m_s2 * (1 + outward @ heading) / 2,
            0.0
        )
        climbing = (
            -self.stiffness_per_s2 * (observation.position[2] - self.height_m)
            - 2 * sqrt(self.stiffness_per_s2) * observation.velocity[2]
        )
        return append(turn * where(outward @ left > 0, -left, left), climbing)

    def scaled(self, factor: float) -> Self:
        """
        Divides the turn by `factor` and the stiffness by its square, as
        a flock magnified by `factor` at the same speed takes, leaving the
        roost's place and size, which the domain sets.
        """
        return tree_at(
            lambda roost: (roost.stiffness_per_s2, roost.turning_m_s2),
            self,
            (self.stiffness_per_s2 / factor**2, self.turning_m_s2 / factor)
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

    amplitude: NonNegative = 2.82
    """
    The amplitude, in meters per second per root second, at a starling's
    scale, by default the noise that leaves a flock aligning at
    `Constant`'s default rate over 7 neighbors at the product of coupling
    and interaction range of 503 inferred from starling flocks,
    `10 * sqrt(2 * 7 * 2.857 / 503)` at 10 meters per second.
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
