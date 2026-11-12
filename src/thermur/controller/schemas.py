"""
Holds the records of the numerical core, meaning the `Flock` a flight
steps, the `Observation` that is the one input a slot reads, the abstract
`Component` filling a slot with the `Heading`, `Noise`, and `Speed` bases,
and the `Composite` summing the slots the settings nest by name.
"""

from ast           import AnnAssign, Constant, Expr, parse
from dataclasses   import fields
from equinox       import Module, tree_at
from inspect       import cleandoc, getsource
from jax           import Array, ShapeDtypeStruct, pure_callback, tree, vmap
from jax.numpy     import cross, einsum, finfo, full, linalg, maximum, sqrt, take, zeros_like
from lineax        import FunctionLinearOperator
from numpy         import arange, argsort, asarray, ndarray, newaxis, stack, take_along_axis
from scipy.spatial import KDTree
from textwrap      import dedent
from typing        import ClassVar, Self

from thermur.environment.air   import Air
from thermur.pipeline.settings import Settings


class Flock(Module):
    """
    The state of every drone, each array laid out as `(..., n_drones, 3)`,
    so a property reads one flock, a flight's frames, or a batch of seeds
    alike.
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

    spins: Array
    """
    Each drone's spin, the angular velocity at which its velocity turns, in
    radians per second, zero under a heading slot holding none.
    """

    velocities: Array
    """
    Each drone's velocity, in meters per second.
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

    @property
    def shapes(self) -> Self:
        """
        Lays out the shape and the type of each array, the structure a
        Brownian path over the flock or an operator onto it reads.
        """
        return tree.map(lambda leaf: ShapeDtypeStruct(leaf.shape, leaf.dtype), self)

    def cruising(self, speed_m_s: float | Array) -> Self:
        """
        Scales each drone's velocity to `speed_m_s`, one speed for every
        drone or one per drone as `(..., n_drones, 1)`, keeping its heading.
        """
        return tree_at(lambda flock: flock.velocities, self, speed_m_s * self.headings)


class Observation(Module):
    """
    What each drone senses at one step, meaning its own state, the state of
    its topological neighbors, and the air at its position, each array laid
    out as `(n_drones, ...)` for a flock and without that axis once `vmap`
    hands one drone's view to a slot, so a slot holds no route to a drone
    its neighbors leave out.
    """

    air: dict[str, Array]
    """
    Each channel of the air at the drone's position, by the channel's name.
    """

    cruise: Array
    """
    The speed the flight sets each drone to cruise at, in meters per second.
    """

    delayed: Array
    """
    Each neighbor's velocity as `Heading.lag_s` earlier, `(n_neighbors, 3)`, in
    meters per second, in the order of its rank.
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
    Each neighbor's velocity, `(n_neighbors, 3)`, in meters per second, in the
    order of its rank.
    """

    offsets: Array
    """
    The offset of each neighbor from the drone, `(n_neighbors, 3)`, in meters,
    in the order of its rank.
    """

    spin: Array
    """
    The drone's own spin, in radians per second.
    """

    velocity: Array
    """
    The drone's own velocity, in meters per second.
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

    @classmethod
    def sensed(  # prose: ignore[unsorted-positionals]
        cls,
        flock  : Flock,
        past   : Array,
        air    : Air,
        t      : Array,
        count  : int,
        cruise : float
    ) -> Self:
        """
        Builds what every drone senses from its `count` nearest
        neighbors, queried on the host through `jax.pure_callback`, whose
        `vmap_method="expand_dims"` sends a batch of seeds in one call.

        Args:
            flock  : The state each drone senses.
            past   : Each drone's velocity at the delay a heading reads,
                     `(n_drones, 3)`.
            air    : The channels each drone reads at its position.
            t      : The time, in seconds since the simulation started.
            count  : The number of neighbors each drone senses.
            cruise : The speed each drone cruises at, in meters per second.

        Returns:
            The observation of the flock, one row per drone.
        """
        positions = flock.positions
        indices   = pure_callback(
            lambda points: nearest(points, count),
            ShapeDtypeStruct((positions.shape[0], count), "int32"),
            positions,
            vmap_method = "expand_dims"
        )
        offsets = take(positions, indices, axis=0) - positions[:, None]
        return cls(
            air       = air.sampled(positions, t),
            cruise    = full(positions.shape[0], cruise),
            delayed   = take(past, indices, axis=0),
            distances = linalg.norm(offsets, axis=-1),
            force     = flock.forces,
            neighbors = take(flock.velocities, indices, axis=0),
            offsets   = offsets,
            spin      = flock.spins,
            velocity  = flock.velocities,
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


class Heading(Component):
    """
    The slot turning a drone toward its neighbors, whose call returns the
    pull of its neighbors in meters per second squared.
    """

    lag_s: ClassVar[float] = 0.0
    """
    The age of the neighbor states the slot reads, in seconds.
    """

    steering: ClassVar[bool] = True
    """
    Whether the pull only turns the drone, leaving its speed to the speed
    slot.
    """

    def kicked(self, increment: Array, observation: Observation) -> tuple[Array, Array]:
        """
        Passes a Brownian increment of the drone's velocity, across it, to
        the velocity whole, leaving the spin unchanged.

        Args:
            increment   : The increment of the velocity across it, in meters
                          per second.
            observation : What the drone senses.

        Returns:
            The increment of the drone's velocity, in meters per second, and
            of its spin, in radians per second.
        """
        return increment, zeros_like(observation.spin)

    def steered(self, force: Array, observation: Observation) -> tuple[Array, Array]:
        """
        Adds `force` to the pull, keeping the part of the sum across the
        drone's velocity where the slot steers and otherwise the part of
        `force` alone, holding the spin at zero.

        Args:
            force       : The other slots' acceleration on the drone, in
                          meters per second squared.
            observation : What the drone senses, its weights set.

        Returns:
            The drone's acceleration, in meters per second squared, and the
            rate of change of its spin, in radians per second squared.
        """
        pull     = self(observation)
        velocity = observation.velocity
        return (
            across(pull + force, velocity) if self.steering
            else pull + across(force, velocity),
            zeros_like(observation.spin)
        )


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
            The amplitude on the velocity, in meters per second per root
            second, and on the force, in meters per second squared per root
            second.
        """
        raise NotImplementedError

    def relaxation(self, observation: Observation) -> Array:
        """
        Holds the force on one drone, in meters per second cubed.
        """
        return zeros_like(observation.force)


class Speed(Component):
    """
    The slot setting a drone's acceleration along its velocity, returning it
    in meters per second squared.
    """

    fixed: ClassVar[bool] = False
    """
    Whether a flight holds each drone at the cruise speed.
    """

    def kicked(self, increment: Array, observation: Observation) -> Array:
        """
        Scales the part of a unit Brownian increment along the drone's
        velocity into the increment of its speed, in meters per second.
        """
        raise NotImplementedError


class Composite(Module):
    """
    The slots the settings nest by name, summed for every drone.
    """

    cohesion: Component
    """
    The acceleration holding a drone at a spacing from its neighbors.
    """

    contact: Component
    """
    The acceleration pushing apart two drones that overlap.
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
    The random increments driving each drone's velocity and force.
    """

    speed: Speed
    """
    The acceleration along a drone's velocity.
    """

    def __call__(self, observation: Observation) -> tuple[Array, Array]:
        """
        Turns the cohesion, the contact, and the noise through the heading,
        which reads the weights the coupling sets, then adds the speed
        slot's acceleration, or keeps only the part across the drone's
        velocity where the speed slot holds the speed fixed.

        Returns:
            The drone's acceleration, in meters per second squared, and the
            rate of change of its spin, in radians per second squared.
        """
        weighted = tree_at(
            lambda sensed: sensed.weights,
            observation,
            self.coupling(observation)
        )
        turned, spun = self.heading.steered(
            self.cohesion(observation)
            + self.contact(observation)
            + self.noise(observation),
            weighted
        )
        return (
            across(turned, observation.velocity) if self.speed.fixed
            else turned + self.speed(weighted),
            spun
        )

    def diffusion(
        self,
        _           : Array,
        flock       : Flock,
        observation : Observation
    ) -> FunctionLinearOperator:
        """
        Maps the Brownian increments onto every drone's state, none reaching
        a position, as the operator a diffrax `ControlTerm` reads.

        The noise slot scales the increments of the force and the part
        of each velocity increment across the velocity, which the heading
        splits between the velocity and the spin, and the speed slot scales
        the part along it.
        """
        velocity, force = vmap(self.noise.amplitudes)(observation)

        def kicked(increment: Flock) -> Flock:
            """
            Scales and splits one set of increments, one per array of the
            flock.
            """
            velocities, spins = vmap(self.heading.kicked)(
                velocity[:, newaxis]
                * across(increment.velocities, observation.velocity),
                observation
            )
            return Flock(
                forces     = force[:, newaxis] * increment.forces,
                positions  = zeros_like(increment.positions),
                spins      = spins,
                velocities = velocities
                + vmap(self.speed.kicked)(increment.velocities, observation)
            )

        return FunctionLinearOperator(kicked, flock.shapes)

    def drift(self, _: Array, flock: Flock, observation: Observation) -> Flock:
        """
        Computes the rate of change of every drone's state, the vector field
        of a diffrax `ODETerm`.
        """
        accelerations, spun = vmap(self)(observation)
        return Flock(
            forces     = vmap(self.noise.relaxation)(observation),
            positions  = flock.velocities,
            spins      = spun,
            velocities = accelerations
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """
        Reads each slot from the settings field naming it.
        """
        return cls(**{slot.name: getattr(settings, slot.name) for slot in fields(cls)})


def across(accelerations: Array, velocities: Array) -> Array:
    """
    Removes the part of each acceleration along its drone's velocity.
    """
    return accelerations - along(accelerations, velocities)


def along(accelerations: Array, velocities: Array) -> Array:
    """
    Projects each acceleration onto its drone's velocity.
    """
    direction = unit(velocities)
    return (accelerations * direction).sum(-1, keepdims=True) * direction


def nearest(points: Array, count: int) -> ndarray:  # prose: ignore[unsorted-positionals]
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
    return stack(
        [
            take_along_axis(
                row,
                argsort(row == own, axis=1, kind="stable")[:, :count],
                axis = 1
            )
            for row in found
        ]
    ).reshape(*clouds.shape[:-1], count).astype("int32")


def torque(accelerations: Array, velocities: Array) -> Array:
    """
    Computes the angular velocity at which each acceleration turns its
    drone's velocity, the cross product of the velocity and the acceleration
    over the squared speed, a stopped drone turning at none.
    """
    return cross(velocities, accelerations) / maximum(
        (velocities**2).sum(-1, keepdims=True),
        finfo(velocities.dtype).tiny
    )


def unit(vectors: Array) -> Array:
    """
    Divides each vector along its last axis by its length, a zero vector
    staying zero.
    """
    lengths = linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / maximum(lengths, finfo(lengths).tiny)


def documented[Kind: type[Component]](kind: Kind) -> Kind:
    """
    Sets the `doc` of each field `kind` declares to the docstring beneath
    it, which cyclopts renders as the field's flag help and Pydantic as
    its description.

    The docstring is read from the class's source with its indentation
    removed, since docstring_parser 0.18.0, which cyclopts reads a
    dataclass's attribute docstrings through, keeps each one's leading
    newline and indentation, which the help then renders as a code block.
    """
    body = parse(dedent(getsource(kind))).body[0].body
    docs = {
        line.target.id: cleandoc(after.value.value)
        for line, after in zip(body, body[1:], strict=False)
        if isinstance(line, AnnAssign)
        and isinstance(after, Expr)
        and isinstance(after.value, Constant)
    }
    for declared in fields(kind):
        declared.doc = docs.get(declared.name, declared.doc)

    return kind
