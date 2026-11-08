"""
Holds the components filling each slot of the controller, each borrowed from
another physical system and cited on the page that explains the choice among
them, meaning the headings, the couplings, the noises, the cohesions, and
the one contact.
"""

from jax        import Array
from jax.numpy  import (
    asarray,
    exp,
    full,
    linalg,
    maximum,
    sqrt,
    square,
    where,
    zeros_like
)
from typing   import ClassVar

from thermur.controller.schemas  import Component, Heading, Noise, Observation, documented, unit
from thermur.environment.schemas import NonNegative, Positive


@documented
class Anisotropic(Component):
    """
    The coupling strengthening a neighbor flying beside a drone over one
    ahead of or behind it, from the anisotropy of jackdaw interactions.
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

    coupling_per_s: NonNegative = 0.6
    """
    The rate at which a drone aligns with each neighbor, in reciprocal
    seconds.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Gives each neighbor the same weight.
        """
        return full(observation.distances.shape, self.coupling_per_s)


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
        velocity     = sqrt(
            square(self.spontaneous) + self.copying_per_s * disagreement.mean() / 3
        )
        return velocity, zeros_like(velocity)


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
class Morse(Component):
    """
    The generalized Morse cohesion over the topological neighbors,
    a repulsion against a longer-range attraction, each decaying
    exponentially.

    Its parameters stay where the potential summed over every pair
    gives a swarm that keeps its spacing as it grows, which holds where
    `repulsion_m2_s2 * repulsion_m**3` exceeds `attraction_m2_s2 *
    attraction_m**3`.
    """

    attraction_m: Positive = 12.0
    """
    The decay length of the attraction, in meters.
    """

    attraction_m2_s2: NonNegative = 80.0
    """
    The strength of the attraction, in square meters per square second.
    """

    repulsion_m: Positive = 2.4
    """
    The decay length of the repulsion, in meters.
    """

    repulsion_m2_s2: NonNegative = 10400.0
    """
    The strength of the repulsion, in square meters per square second.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the force of the potential's gradient along each bearing.
        """
        distances = observation.distances
        pull      = (
            self.attraction_m2_s2 / self.attraction_m * exp(
                -distances / self.attraction_m
            )
            - self.repulsion_m2_s2 / self.repulsion_m * exp(
                -distances / self.repulsion_m
            )
        )
        return pull @ observation.bearings

    def __check_init__(self):
        """
        Refuses parameters whose repulsion, times the cube of its length,
        does not exceed the attraction's.

        Raises:
            ValueError: Where the parameters sit in the region where a swarm
                        summing the potential over every pair collapses.
        """
        repelling  = self.repulsion_m2_s2 * self.repulsion_m**3
        attracting = self.attraction_m2_s2 * self.attraction_m**3
        if repelling <= attracting:
            raise ValueError(
                f"The repulsion's {repelling:g} m^5/s^2 does not exceed the "
                f"attraction's {attracting:g} m^5/s^2, where a swarm collapses"
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
class SoftCore(Component):
    """
    The harmonic soft-sphere contact, whose force grows linearly with
    overlap and stays bounded by `stiffness_m_s2` however far two drones
    overlap.
    """

    diameter_m: Positive = 6.0
    """
    The distance within which two drones push apart, in meters.
    """

    stiffness_m_s2: NonNegative = 8.0
    """
    The acceleration at full overlap, in meters per second squared.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the push from each overlapping neighbor against its bearing.
        """
        overlap = maximum(1 - observation.distances / self.diameter_m, 0.0)
        return -self.stiffness_m_s2 * overlap @ observation.bearings


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
        order    = linalg.norm(unit(observation.neighbors).mean(0))
        velocity = self.amplitude / maximum(order, self.floor)
        return velocity, zeros_like(velocity)


@documented
class Weighted(Component):
    """
    The coupling decaying with distance even within a rank, the rule chimney
    swifts show.
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
    velocity.
    """

    amplitude: NonNegative = 4.2
    """
    The amplitude, in meters per second per root second.
    """

    def amplitudes(self, observation: Observation) -> tuple[Array, Array]:
        """
        Drives every velocity at one amplitude, leaving the force undriven.
        """
        velocity = asarray(self.amplitude)
        return velocity, zeros_like(velocity)
