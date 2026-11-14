"""
Holds the records of the numerical core, meaning the `Flock` a flight steps,
the `Radio` carrying each drone's state to the others and the `Inbox` it
keeps, the `Observation` that is the one input a slot reads, the abstract
`Component` filling a slot, and the `Composite` summing the slots the
settings nest by name.
"""

from dataclasses import fields
from equinox     import Module, field, tree_at
from jax         import Array, lax, tree, vmap
from jax.numpy     import (
    broadcast_to,
    einsum,
    expand_dims,
    eye,
    finfo,
    isfinite,
    linalg,
    maximum,
    sqrt,
    stack,
    where,
    zeros_like
)
from jax.random    import bernoulli, fold_in
from lineax        import DiagonalLinearOperator
from math          import inf
from numpy         import arange, argsort, asarray, ndarray, newaxis, take_along_axis
from pydantic      import Field
from scipy.spatial import KDTree
from typing        import Annotated, ClassVar, Self

from thermur.environment.air     import Air
from thermur.environment.plant   import Plant
from thermur.environment.schemas import NonNegative, Positive, documented
from thermur.pipeline.settings   import Settings


class Flock(Module):
    """
    The state of every drone, each array laid out as `(..., n_drones, 3)`,
    so a property reads one flock, a flight's frames, or a batch of seeds
    alike.
    """

    commands: Array
    """
    The velocity each drone's controller commands, in meters per second,
    which its own velocity follows through the plant.
    """

    forces: Array
    """
    The persistent random force on each drone, in meters per second squared,
    zero under a noise slot holding none.
    """

    positions: Array
    """
    Each drone's east, north, and height above the ground, in meters.
    """

    velocities: Array
    """
    Each drone's velocity through the air, in meters per second.
    """

    @property
    def aspect_ratios(self) -> Array:
        """
        Reads the flock's shape from the eigenvalues of `gyration`, in
        ascending order, as the length of each axis over the thinnest's.
        """
        lengths = sqrt(linalg.eigvalsh(self.gyration))
        return lengths / lengths[..., :1]

    @property
    def gyration(self) -> Array:
        """
        Computes the gyration tensor, the mean outer product of each drone's
        offset from the flock's center, `(..., 3, 3)`, in square meters.
        """
        offsets = self.positions - self.positions.mean(-2, keepdims=True)
        return einsum("...ni,...nj->...ij", offsets, offsets) / offsets.shape[-2]

    @property
    def headings(self) -> Array:
        """
        Divides each drone's velocity by its speed.
        """
        return unit(self.velocities)

    @property
    def polarization(self) -> Array:
        """
        Computes the length of the mean heading, from zero for headings that
        cancel to one for a flock flying as one.
        """
        return linalg.norm(self.headings.mean(-2), axis=-1)

    def cruising(self, speed_m_s: float) -> Self:
        """
        Scales each drone's command to `speed_m_s`, keeping its direction.
        """
        return tree_at(
            lambda flock: flock.commands,
            self,
            speed_m_s * unit(self.commands)
        )


class Inbox(Module):
    """
    The state of each drone as each other drone last heard it over the
    radio, each array laid out as `(n_receivers, n_senders, ...)`.
    """

    delayed: Array
    """
    Each sender's velocity `Heading.lag_s` before the broadcast each receiver
    last heard, in meters per second.
    """

    heard: Array
    """
    Whether each receiver holds any broadcast of each sender.
    """

    positions: Array
    """
    Each sender's position in the broadcast each receiver last heard, in
    meters.
    """

    velocities: Array
    """
    Each sender's velocity in the broadcast each receiver last heard, in
    meters per second.
    """

    @classmethod
    def from_broadcast(cls, earlier: Flock, range_m: float, sent: Flock) -> Self:
        """
        Builds the inbox in which every drone last heard the broadcast
        `sent`, each sender's velocity `Heading.lag_s` before it read from
        `earlier`, each receiver hearing only the senders within `range_m`
        of it as `sent` went out.
        """
        drones = sent.positions.shape[-2]
        shape  = (drones, drones, 3)
        return cls(
            delayed = broadcast_to(earlier.velocities, shape),
            heard   = linalg.norm(
                sent.positions[:, newaxis] - sent.positions,
                axis = -1
            ) <= range_m,
            positions  = broadcast_to(sent.positions, shape),
            velocities = broadcast_to(sent.velocities, shape)
        )


@documented
class Radio(Module):
    """
    The link each drone broadcasts its position and velocity over, each
    packet reaching another drone within range late, at a fixed rate, and
    only where the link does not lose it, after the simulated radios of an
    outdoor drone flock.
    """

    delay_s: NonNegative = field(default=1.0, static=True)
    """
    The age of a broadcast as it arrives, in seconds.
    """

    loss: Annotated[float, Field(ge=0, lt=1)] = 0.4
    """
    The share of the packets between two drones the link loses, each loss
    drawn apart from every other.
    """

    range_m: Positive = field(default=50.0, static=True)
    """
    The distance between two drones, as a packet is sent, past which the
    packet never arrives, in meters, by default the distance past which the
    outdoor drone flock's radios began to fail.
    """

    rate_hz: Positive = field(default=5.0, static=True)
    """
    The number of packets each drone sends each second.
    """

    def delay_steps(self, step_s: float) -> int:
        """
        Counts the steps of `step_s` that `delay_s` spans.
        """
        return round(self.delay_s / step_s)

    def heard(  # prose: ignore[unsorted-positionals]
        self,
        inbox  : Inbox,
        sent   : Flock,
        flock  : Flock,
        key    : Array,
        step   : Array,
        lag    : int,
        step_s : float
    ) -> tuple[Inbox, Flock]:
        """
        Keeps the broadcast `flock` sends at `step`, then, where a packet
        falls at `step`, delivers to each drone the broadcast each other
        drone sent `delay_s` earlier from within `range_m` of it, each drone
        holding what it last heard of a sender whose packet the link loses
        or whose distance kept it from arriving.

        Args:
            inbox  : What each drone heard by the step before.
            sent   : The flock at each of the last steps the radio keeps,
                     each array laid out as `(n_kept, n_drones, 3)`, the
                     step `s` under the index `s % n_kept`.
            flock  : The state each drone broadcasts.
            key    : The key each packet's loss derives from.
            step   : The number of steps since launch.
            lag    : The number of steps `Heading.lag_s` spans.
            step_s : The step of the integrator, in seconds.

        Returns:
            What each drone heard by `step`, and the broadcasts kept with
            `flock`'s among them.
        """
        kept = sent.positions.shape[0]
        sent = tree.map(
            lambda history, now: history.at[step % kept].set(now),
            sent,
            flock
        )
        late  = step - self.delay_steps(step_s)
        fresh = Inbox.from_broadcast(
            earlier = tree.map(lambda history: history[(late - lag) % kept], sent),
            range_m = self.range_m,
            sent    = tree.map(lambda history: history[late % kept], sent)
        )
        carried = fresh.heard & bernoulli(
            fold_in(key, step),
            1 - self.loss,
            fresh.heard.shape
        ) & (step % self.period_steps(step_s) == 0)
        return tree.map(
            lambda new, old: where(
                expand_dims(carried, tuple(range(carried.ndim, new.ndim))),
                new,
                old
            ),
            fresh,
            inbox
        ), sent

    def opened(self, flock: Flock, lag: int, step_s: float) -> tuple[Inbox, Flock]:
        """
        Builds what the radio holds at launch, every drone having heard the
        launch state of each other drone within `range_m` of it, and every
        broadcast it keeps being the launch's, enough of them to deliver one
        `delay_s` late and its velocity `lag` steps before that.

        Returns:
            What each drone heard at launch, and the broadcasts kept.
        """
        kept = self.delay_steps(step_s) + lag + 1
        return Inbox.from_broadcast(flock, self.range_m, flock), tree.map(
            lambda leaf: broadcast_to(leaf, (kept, *leaf.shape)),
            flock
        )

    def period_steps(self, step_s: float) -> int:
        """
        Counts the steps of `step_s` between two packets, at least one.
        """
        return max(round(1 / (self.rate_hz * step_s)), 1)

    def scaled(self, _: float) -> Self:
        """
        Leaves the radio as it stands, since its delay, its rate, its loss,
        and its range are a drone radio's own at any spacing.
        """
        return self


class Observation(Module):
    """
    What each drone senses at one step, meaning its own state, the state
    of its topological neighbors as its radio last heard it, and the air at
    its position, each array laid out as `(n_drones, ...)` for a flock and
    without that axis once `vmap` hands one drone's view to a slot, so a
    slot holds no route to a drone its neighbors leave out.
    """

    air: dict[str, Array]
    """
    Each channel of the air at the drone's position, by the channel's name.
    """

    delayed: Array
    """
    Each neighbor's velocity `Heading.lag_s` before the state its radio last
    heard, `(n_neighbors, 3)`, in meters per second, in the order of its rank.
    """

    distances: Array
    """
    The distance to each neighbor, `(n_neighbors,)`, in meters, in the order
    of its rank.
    """

    force: Array
    """
    The persistent random force on the drone, in meters per second squared.
    """

    neighbors: Array
    """
    Each neighbor's velocity as its radio last heard it, `(n_neighbors, 3)`,
    in meters per second, in the order of its rank.
    """

    offsets: Array
    """
    The offset of each neighbor from the drone, `(n_neighbors, 3)`, in meters,
    in the order of its rank.
    """

    position: Array
    """
    The drone's own east, north, and height above the ground, in meters.
    """

    velocity: Array
    """
    The velocity the drone commands, in meters per second.
    """

    weights: Array
    """
    The rate at which the drone aligns with each neighbor, `(n_neighbors,)`,
    in reciprocal seconds, which `Composite` sets from its coupling slot.
    """

    @property
    def ahead(self) -> Array:
        """
        Computes the cosine of each neighbor's bearing off the drone's
        velocity, one for a neighbor straight ahead.
        """
        return self.bearings @ unit(self.velocity)

    @property
    def bearings(self) -> Array:
        """
        Divides each offset by its distance, a neighbor sitting on the drone
        taking a bearing of zero.
        """
        return self.offsets / maximum(
            self.distances,
            finfo(self.distances).tiny
        )[:, None]

    @property
    def wind(self) -> Array:
        """
        Gathers the east, north, and upward wind at the drone's position, in
        meters per second.
        """
        return stack([self.air[name] for name in ("u", "v", "w")], axis=-1)

    @classmethod
    def sensed(  # prose: ignore[unsorted-positionals]
        cls,
        flock : Flock,
        inbox : Inbox,
        air   : Air,
        t     : Array,
        count : int
    ) -> Self:
        """
        Builds what every drone senses from its `count` nearest neighbors,
        ranked by the positions it heard among the senders `inbox` holds a
        broadcast of, the radio's range having acted on the true distance as
        each packet was sent.

        A rank no heard sender fills reads as a neighbor sitting on the
        drone and flying its command, so it adds nothing to a sum of offsets
        or of differences from the command, though it still counts toward a
        mean over the neighbors.

        Args:
            flock : The state each drone senses.
            inbox : The states each drone last heard of every other.
            air   : The channels each drone reads at its position.
            t     : The time, in seconds since the simulation started.
            count : The number of neighbors each drone senses.

        Returns:
            The observation of the flock, one row per drone.
        """
        positions       = flock.positions
        drones          = positions.shape[0]
        nearer, indices = lax.top_k(
            -where(
                inbox.heard & ~eye(drones, dtype=bool),
                linalg.norm(inbox.positions - positions[:, newaxis], axis=-1),
                inf
            ),
            count
        )
        rows    = arange(drones)[:, newaxis]
        found   = isfinite(nearer)[..., newaxis]
        senders = where(found[..., 0], indices, rows)

        def told(held: Array, own: Array) -> Array:
            """
            Reads what each drone heard of each neighbor in `held`, or `own`
            where no heard sender fills the rank.
            """
            return where(found, held[rows, senders], own[:, newaxis])

        offsets = told(inbox.positions, positions) - positions[:, newaxis]
        return cls(
            air       = air.sampled(positions, t),
            delayed   = told(inbox.delayed, flock.commands),
            distances = linalg.norm(offsets, axis=-1),
            force     = flock.forces,
            neighbors = told(inbox.velocities, flock.commands),
            offsets   = offsets,
            position  = positions,
            velocity  = flock.commands,
            weights   = zeros_like(indices, dtype="float32")
        )


class Component(Module):
    """
    One slot of the controller, whose fields are its parameters and whose
    call reads one drone's `Observation`, so it holds no route to any drone
    beyond that drone's neighbors.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Computes what the slot contributes for one drone, in the unit each
        slot's own base names.
        """
        raise NotImplementedError

    def scaled(self, factor: float) -> Self:
        """
        Carries the slot from a starling flock to one magnified by `factor`
        at the same speed, leaving a slot whose values are a drone flock's
        own as it stands.
        """
        return self


class Heading(Component):
    """
    The slot turning a drone toward its neighbors, returning an acceleration
    in meters per second squared.
    """

    confined: ClassVar[bool] = True
    """
    Whether a flight holds each drone at the cruise speed.
    """

    lag_s: ClassVar[float] = 0.0
    """
    The age of the neighbor states the slot reads, in seconds.
    """


class Noise(Component):
    """
    The slot driving each drone with a Brownian increment, returning the
    acceleration its state adds, zero for a slot holding no state.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Adds no acceleration beside the increment `amplitudes` scales.
        """
        return zeros_like(observation.velocity)

    def amplitudes(self, observation: Observation) -> tuple[Array, Array]:
        """
        Computes what scales one drone's Brownian increments.

        Returns:
            The amplitude on the command, in meters per second per root
            second, and on the force, in meters per second squared per root
            second.
        """
        raise NotImplementedError

    def relaxation(self, observation: Observation) -> Array:
        """
        Holds the force on one drone, in meters per second cubed.
        """
        return zeros_like(observation.force)


class Composite(Module):
    """
    The slots the settings nest by name, summed for every drone into the
    rate of its command, beside the radio carrying each drone's state to the
    others and the plant carrying each drone's velocity toward its command.
    """

    arena: Component
    """
    The acceleration turning a drone back from the walls of the box it
    flies in.
    """

    cohesion: Component
    """
    The acceleration holding a drone at a spacing from its neighbors.
    """

    contact: Component
    """
    The acceleration pushing a drone away from each of its neighbors.
    """

    coupling: Component
    """
    The rate at which a drone aligns with each neighbor.
    """

    heading: Heading
    """
    The acceleration turning a drone toward its neighbors.
    """

    noise: Noise
    """
    The random increments driving each drone's command and force.
    """

    plant: Plant
    """
    The lag and the bounds carrying a drone's velocity toward its command.
    """

    radio: Radio
    """
    The link carrying each drone's state to the others.
    """

    roost: Component
    """
    The acceleration pulling a drone toward the roost's height.
    """

    speed: Component
    """
    The acceleration relaxing a drone's speed toward the cruise speed.
    """

    def __call__(self, observation: Observation) -> Array:
        """
        Sums the slots for one drone, the heading reading the weights the
        coupling sets, and takes the part across the drone's command where
        the heading holds the speed fixed.

        Returns:
            The rate of the drone's command, in meters per second squared.
        """
        weighted = tree_at(
            lambda sensed: sensed.weights,
            observation,
            self.coupling(observation)
        )
        total = sum(
            (
                self.arena(observation),
                self.cohesion(observation),
                self.contact(observation),
                self.heading(weighted),
                self.noise(observation),
                self.roost(observation),
                self.speed(observation)
            )
        )
        return across(total, observation.velocity) if self.heading.confined else total

    def diffusion(
        self,
        _           : Array,
        flock       : Flock,
        observation : Observation
    ) -> DiagonalLinearOperator:
        """
        Lays out the amplitude of every Brownian increment, each driving
        a command or a force and none a position or a velocity, as the
        diagonal operator a diffrax `ControlTerm` reads.
        """
        command, force = vmap(self.noise.amplitudes)(observation)
        return DiagonalLinearOperator(
            Flock(
                commands   = broadcast_to(command[:, newaxis], flock.commands.shape),
                forces     = broadcast_to(force[:, newaxis], flock.forces.shape),
                positions  = zeros_like(flock.positions),
                velocities = zeros_like(flock.velocities)
            )
        )

    def drift(self, _: Array, flock: Flock, observation: Observation) -> Flock:
        """
        Computes the rate of change of every drone's state, the vector field
        of a diffrax `ODETerm`, each drone moving over the ground at its
        velocity through the air plus the wind at its position.
        """
        return Flock(
            commands   = vmap(self)(observation),
            forces     = vmap(self.noise.relaxation)(observation),
            positions  = flock.velocities + observation.wind,
            velocities = self.plant.tracked(flock.commands, flock.velocities)
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """
        Reads each slot from the settings field naming it, scaled from a
        starling flock to the plan's launch by `Plan.similarity`.
        """
        return cls(
            **{
                slot.name: getattr(settings, slot.name).scaled(settings.plan.similarity)
                for slot in fields(cls)
            }
        )


def across(accelerations: Array, velocities: Array) -> Array:
    """
    Removes the part of each acceleration along its drone's velocity.
    """
    direction = unit(velocities)
    along     = (accelerations * direction).sum(-1, keepdims=True)
    return accelerations - along * direction


def nearest(  # prose: ignore[unsorted-positionals]
    points : Array,
    count  : int
) -> ndarray:
    """
    Queries each drone's `count` nearest other drones on the host, leaving
    a drone out of its own row even where others sit at its exact position,
    which `KDTree.query` may order ahead of it.

    Args:
        points : The positions, `(..., n_drones, 3)`.
        count  : The number of neighbors per drone.

    Returns:
        The indices of each drone's neighbors in the order of their distance,
        `(..., n_drones, count)`.
    """
    clouds = asarray(points)
    drones = clouds.shape[-2]
    own    = arange(drones)[:, newaxis]
    found  = [
        KDTree(cloud).query(cloud, k=count + 1)[1]
        for cloud in clouds.reshape(-1, drones, 3)
    ]
    return asarray(
        [
            take_along_axis(
                row,
                argsort(row == own, axis=1, kind="stable")[:, :count],
                axis = 1
            )
            for row in found
        ]
    ).reshape(*clouds.shape[:-1], count).astype("int32")


def unit(vectors: Array) -> Array:
    """
    Divides each vector along its last axis by its length, a zero vector
    staying zero.
    """
    lengths = linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / maximum(lengths, finfo(lengths).tiny)
