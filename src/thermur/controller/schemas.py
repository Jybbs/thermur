"""
Holds the records of the numerical core, meaning the `Flock` a flight steps,
the `Radio` carrying each drone's state to the others and the `Inbox` it
keeps, the `Observation` that is the one input a slot reads, the abstract
`Component` filling a slot with the `Heading`, `Noise`, and `Speed` bases,
and the `Composite` summing the slots the settings nest by name.
"""

from dataclasses import fields
from equinox     import Module, field, filter_jit, tree_at
from functools   import partial
from jax         import Array, ShapeDtypeStruct, lax, tree, vmap
from jax.numpy     import (
    broadcast_to,
    concatenate,
    cross,
    einsum,
    eye,
    finfo,
    full,
    isfinite,
    linalg,
    maximum,
    sqrt,
    stack,
    unique,
    vectorize,
    where,
    zeros_like
)
from jax.random    import fold_in, uniform
from lineax        import FunctionLinearOperator
from math          import inf
from numpy         import arange, argsort, asarray, ndarray, newaxis, take_along_axis
from pydantic      import Field
from scipy.spatial import KDTree
from typing        import Annotated, ClassVar, Self

from thermur.environment.air     import Air
from thermur.environment.plant   import Plant
from thermur.environment.schemas import Count, NonNegative, Positive, documented
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
    The state the noise slot holds on each drone, in the units that slot
    names, zero under a slot holding none.
    """

    positions: Array
    """
    Each drone's east, north, and height above the ground, in meters.
    """

    spins: Array
    """
    Each drone's spin, the angular velocity at which its command turns, in
    radians per second, zero under a heading slot holding none.
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

    @property
    def shapes(self) -> Self:
        """
        Lays out the shape and the type of each array, the structure a
        Brownian path over the flock or an operator onto it reads.
        """
        return tree.map(lambda leaf: ShapeDtypeStruct(leaf.shape, leaf.dtype), self)

    def cruising(self, speed_m_s: float | Array) -> Self:
        """
        Scales each drone's command to `speed_m_s`, one speed for every
        drone or one per drone as `(..., n_drones, 1)`, keeping its
        direction.
        """
        return tree_at(
            lambda flock: flock.commands,
            self,
            speed_m_s * unit(self.commands)
        )


class Inbox(Module):
    """
    The state of each sender a drone holds in its table as the drone last
    heard it over the radio, each array laid out as `(n_receivers, n_slots,
    ...)`, a slot holding no sender naming the receiver itself.
    """

    delayed: Array
    """
    Each held sender's velocity `Heading.lag_s` before the broadcast the
    receiver last heard, in meters per second.
    """

    forces: Array
    """
    Each held sender's noise state in the broadcast the receiver last heard.
    """

    heard: Array
    """
    Whether each slot holds a broadcast of its sender younger than the
    radio's hold.
    """

    lists: Array
    """
    The senders each drone's table held at each of the last steps the radio
    keeps, `(n_kept, n_drones, n_slots)`, the neighbor list each broadcast
    carries, the step `s` under the index `s % n_kept`.
    """

    positions: Array
    """
    Each held sender's position in the broadcast the receiver last heard, in
    meters.
    """

    received: Array
    """
    The step at which the receiver last took a packet from each held sender.
    """

    senders: Array
    """
    The drone each slot holds.
    """

    velocities: Array
    """
    Each held sender's velocity in the broadcast the receiver last heard, in
    meters per second.
    """

    @classmethod
    def from_broadcast(
        cls,
        earlier : Flock,
        range_m : float,
        sent    : Flock,
        slots   : int | None = None
    ) -> Self:
        """
        Builds the inbox in which every drone last heard the broadcast
        `sent` from its `slots` nearest other drones within `range_m` of
        it as `sent` went out, by default the radio's own `slots`, each
        sender's velocity `Heading.lag_s` before it read from `earlier`, and
        each packet taken at step zero.

        The search compares every pair of drones once, and a flight runs it
        only as it opens or replays a frame.
        """
        slots    = slots or Radio.slots
        drones   = sent.positions.shape[-2]
        apart    = linalg.norm(sent.positions[:, newaxis] - sent.positions, axis=-1)
        distance = where((apart <= range_m) & ~eye(drones, dtype=bool), apart, inf)
        indices, heard = closest(
            count     = slots,
            distances = concatenate(
                [distance, full((drones, max(slots - drones, 0)), inf)],
                1
            )
        )
        senders = where(heard, indices, arange(drones)[:, newaxis])
        return cls(
            delayed    = earlier.velocities[senders],
            forces     = sent.forces[senders],
            heard      = heard,
            lists      = senders[newaxis],
            positions  = sent.positions[senders],
            received   = zeros_like(senders),
            senders    = senders,
            velocities = sent.velocities[senders]
        )


@documented
class Radio(Module):
    """
    The link each drone broadcasts its position, its velocity, its noise
    state, and its table's senders over, each packet reaching another drone
    within range late, at a fixed rate, and only where the link does not
    lose it, after the simulated radios of an outdoor drone flock, each
    receiver forgetting a sender once `hold_periods` packet periods pass
    with no packet from it, the neighbor hold time of the mobile ad hoc
    routing protocols.

    Each receiver keeps in its table the `slots` nearest senders it hears,
    and learns a sender beyond its table in two ways:

    - The sender appears in the neighbor list a member of its table
      broadcasts, as OLSR populates its 2-hop neighbor set in section 8.2
      of RFC 3626
    - The sender is another drone in the `slots` places along the Z-order
      curve of the broadcasts that start `slots / 2` before its own, which
      reaches a drone whose members list no sender in range
    """

    delay_s: NonNegative = field(default=1.0, static=True)
    """
    The age of a broadcast as it arrives, in seconds.
    """

    hold_periods: Count = field(default=3, static=True)
    """
    The number of packet periods after its last packet for which a receiver
    keeps hearing a sender, by default the three refresh intervals the
    mobile ad hoc routing protocols OLSR and NHDP set as the neighbor hold
    time.
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

    slots: ClassVar[int] = 16
    """
    The number of senders each drone's table holds, at least the plan's
    `neighbors`, the smallest of 8, 12, 16, and 24 at which the seven
    nearest senders a table yields matched an exact search over the same
    broadcasts for more than 99.5% of the neighbors, measured at 1,000 and
    3,000 drones whose velocities scatter by 1.5 m/s, after 12 s under the
    default link.
    """

    def delay_steps(self, step_s: float) -> int:
        """
        Counts the steps of `step_s` that `delay_s` spans.
        """
        return round(self.delay_s / step_s)

    def delivered(
        self,
        broadcast : Flock,
        earlier   : Flock,
        flock     : Flock,
        inbox     : Inbox,
        key       : Array,
        listed    : Array,
        step      : Array,
        step_s    : float
    ) -> Inbox:
        """
        Delivers the packet falling at `step` to each receiver from three
        kinds of sender:

        - Each sender its table holds
        - Each sender the neighbor lists of the members it hears name
        - Each other drone in the `slots` places along the Z-order curve
          that start `slots / 2` before its own

        A packet arrives where its sender broadcast from within `range_m`
        of the receiver and the link keeps it. Each receiver then keeps in
        its table the `slots` nearest of these senders and of those it still
        holds, ranked by the positions it heard.

        Args:
            broadcast: The broadcast each drone sent `delay_s` earlier.
            earlier: The flock `Heading.lag_s` before `broadcast`.
            flock: The state each drone now holds.
            inbox: What each drone heard by the step before.
            key: The key each packet's loss derives from.
            listed: The senders each drone's table held as it sent
                    `broadcast`, `(n_drones, n_slots)`.
            step: The number of steps since launch.
            step_s: The step of the integrator, in seconds.

        Returns:
            What each drone heard by `step`.
        """
        drones, slots = inbox.senders.shape
        rows          = arange(drones)[:, newaxis]

        def arrived(senders: Array) -> Array:
            """
            Finds whether the packet from each of `senders` reaches its row,
            a drone never hearing itself.
            """
            sent_m = linalg.norm(
                broadcast.positions[senders] - broadcast.positions[:, newaxis],
                axis = -1
            )
            return (
                (sent_m <= self.range_m)
                & (self.draws(key, rows, senders, step) < 1 - self.loss)
                & (senders != rows)
            )

        curve   = zorder(broadcast.positions)
        members = arrived(inbox.senders)
        found   = concatenate(
            [
                where(members[..., newaxis], listed[inbox.senders], rows[..., newaxis])
                .reshape(drones, -1),
                curve[(curve.argsort()[:, newaxis] + arange(slots) - slots // 2).clip(
                    0,
                    drones - 1
                )]
            ],
            1
        )
        fresh = concatenate([members, arrived(found)], 1)
        alive = fresh.at[:, :slots].set(
            members
            | inbox.heard
            & (step - inbox.received < self.hold_periods * self.period_steps(step_s))
        )
        ids, first = vmap(
            partial(unique, fill_value=drones, return_index=True, size=alive.shape[1])
        )(where(alive, concatenate([inbox.senders, found], 1), drones))
        fresh   = fresh[rows, first]
        slot    = first.clip(0, slots - 1)
        heard_m = linalg.norm(
            where(
                fresh[..., newaxis],
                broadcast.positions[ids],
                inbox.positions[rows, slot]
            ) - flock.positions[:, newaxis],
            axis = -1
        )
        picked, kept = closest(slots, where(ids < drones, heard_m, inf))
        senders      = where(kept, ids[rows, picked], rows)
        taken        = fresh[rows, picked]
        held         = slot[rows, picked]

        def told(new: Array, old: Array) -> Array:
            """
            Reads each kept sender's state from the packet just taken, or
            from the slot it held where no packet came.
            """
            return where(taken[..., newaxis], new[senders], old[rows, held])

        return tree_at(
            lambda table: (
                table.delayed, table.forces, table.heard, table.positions,
                table.received, table.senders, table.velocities
            ),
            inbox,
            (
                told(earlier.velocities, inbox.delayed),
                told(broadcast.forces, inbox.forces),
                kept,
                told(broadcast.positions, inbox.positions),
                where(taken, step, inbox.received[rows, held]),
                senders,
                told(broadcast.velocities, inbox.velocities)
            )
        )

    def draws(
        self,
        key       : Array,
        receivers : Array,
        senders   : Array,
        step      : Array
    ) -> Array:
        """
        Draws the uniform number each packet from `senders` to `receivers`
        at `step` compares with the loss, from a key folded from the step,
        the receiver, and the sender, so one pair at one step draws the same
        number wherever a search finds it.
        """
        stepped = fold_in(key, step)
        return vectorize(
            lambda receiver, sender: uniform(
                fold_in(fold_in(stepped, receiver), sender)
            )
        )(receivers, senders)

    @filter_jit
    def heard(
        self,
        flock  : Flock,
        inbox  : Inbox,
        key    : Array,
        lag    : int,
        sent   : Flock,
        step   : Array,
        step_s : float
    ) -> tuple[Inbox, Flock]:
        """
        Keeps the broadcast `flock` sends at `step` beside each drone's
        table as its neighbor list, then, where a packet falls at `step`,
        delivers the broadcasts sent `delay_s` earlier as `delivered` sets
        out, each drone holding what it last heard of a sender whose packet
        the link loses or whose distance kept it from arriving, and hearing
        that sender until `hold_periods` packet periods pass with no packet
        from it.

        Args:
            flock: The state each drone broadcasts.
            inbox: What each drone heard by the step before.
            key: The key each packet's loss derives from.
            lag: The number of steps `Heading.lag_s` spans.
            sent: The flock at each of the last steps the radio keeps, each
                  array laid out as `(n_kept, n_drones, 3)`, the step `s`
                  under the index `s % n_kept`.
            step: The number of steps since launch.
            step_s: The step of the integrator, in seconds.

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
        listing = tree_at(
            lambda table: table.lists,
            inbox,
            inbox.lists.at[step % kept].set(inbox.senders)
        )
        late = step - self.delay_steps(step_s)
        return lax.cond(
            step % self.period_steps(step_s) == 0,
            lambda: self.delivered(
                broadcast = tree.map(lambda history: history[late % kept], sent),
                earlier = tree.map(lambda history: history[(late - lag) % kept], sent),
                flock   = flock,
                inbox   = listing,
                key     = key,
                listed  = listing.lists[late % kept],
                step    = step,
                step_s  = step_s
            ),
            lambda: listing
        ), sent

    def opened(self, flock: Flock, lag: int, step_s: float) -> tuple[Inbox, Flock]:
        """
        Builds what the radio holds at launch, every drone having heard the
        launch state of its `slots` nearest other drones within `range_m`
        of it, and every broadcast and neighbor list it keeps being the
        launch's, enough of them to deliver one `delay_s` late and its
        velocity `lag` steps before that.

        Returns:
            What each drone heard at launch, and the broadcasts kept.
        """
        kept  = self.delay_steps(step_s) + lag + 1
        inbox = Inbox.from_broadcast(flock, self.range_m, flock, self.slots)
        return tree_at(
            lambda table: table.lists,
            inbox,
            broadcast_to(inbox.lists, (kept, *inbox.lists.shape[1:]))
        ), tree.map(lambda leaf: broadcast_to(leaf, (kept, *leaf.shape)), flock)

    def period_steps(self, step_s: float) -> int:
        """
        Counts the steps of `step_s` between two packets, at least one.
        """
        return max(round(1 / (self.rate_hz * step_s)), 1)

    def scaled(self, _: float) -> Self:
        """
        Leaves the radio as it stands, since its delay, its rate, its loss,
        its range, and its table are a drone radio's own at any spacing.
        """
        return self


class Observation(Module):
    """
    What each drone senses at one step, meaning its own state, the state
    of its topological neighbors as its radio last heard it, and the air at
    its position, each array laid out as `(n_drones, ...)` for a flock and
    without that axis once `vmap` hands one drone's view to a slot, so a
    slot holds no route to a drone its radio does not hear.
    """

    air: dict[str, Array]
    """
    Each channel of the air at the drone's position, by the channel's name.
    """

    centrality: Array
    """
    The length of the mean bearing of every sender the drone's radio holds,
    near zero inside the flock and rising toward one at its border.
    """

    cruise: Array
    """
    The speed the flight sets each drone to cruise at, in meters per second.
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
    The state the noise slot holds on the drone.
    """

    forces: Array
    """
    Each neighbor's noise state as its radio last heard it, `(n_neighbors,
    3)`, in the order of its rank.
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

    spin: Array
    """
    The drone's own spin, the angular velocity at which its command turns,
    in radians per second.
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
    def sensed(
        cls,
        air    : Air,
        count  : int,
        cruise : float,
        flock  : Flock,
        inbox  : Inbox,
        t      : Array
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
            air: The channels each drone reads at its position.
            count: The number of neighbors each drone senses.
            cruise: The speed each drone cruises at, in meters per second.
            flock: The state each drone senses.
            inbox: The states each drone last heard of each sender its table
                   holds.
            t: The time, in seconds since the simulation started.

        Returns:
            The observation of the flock, one row per drone.
        """
        positions       = flock.positions
        drones          = positions.shape[0]
        apart           = inbox.positions - positions[:, newaxis]
        indices, filled = closest(
            count     = count,
            distances = where(inbox.heard, linalg.norm(apart, axis=-1), inf)
        )
        rows    = arange(drones)[:, newaxis]
        found   = filled[..., newaxis]
        senders = where(filled, indices, rows)

        def told(held: Array, own: Array) -> Array:
            """
            Reads what each drone heard of each neighbor in `held`, or `own`
            where no heard sender fills the rank.
            """
            return where(found, held[rows, senders], own[:, newaxis])

        offsets = told(inbox.positions, positions) - positions[:, newaxis]
        return cls(
            air        = air.sampled(positions, t),
            centrality = linalg.norm(
                where(inbox.heard[..., newaxis], unit(apart), 0.0).sum(1),
                axis = -1
            ) / maximum(inbox.heard.sum(1), 1),
            cruise    = full(drones, cruise),
            delayed   = told(inbox.delayed, flock.commands),
            distances = linalg.norm(offsets, axis=-1),
            force     = flock.forces,
            forces    = told(inbox.forces, flock.forces),
            neighbors = told(inbox.velocities, flock.commands),
            offsets   = offsets,
            position  = positions,
            spin      = flock.spins,
            velocity  = flock.commands,
            weights   = zeros_like(indices, dtype="float32")
        )


class Component(Module):
    """
    One slot of the controller, whose fields are its parameters and whose
    call reads one drone's `Observation`, so it holds no route to any drone
    its radio does not hear.
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
    The slot turning a drone toward its neighbors, whose call returns the
    pull of its neighbors in meters per second squared.
    """

    lag_s: ClassVar[float] = 0.0
    """
    The age of the neighbor states the slot reads, in seconds.
    """

    steering: ClassVar[bool] = True
    """
    Whether the pull only turns the drone's command, leaving its speed to
    the speed slot.
    """

    def kicked(self, increment: Array, observation: Observation) -> tuple[Array, Array]:
        """
        Passes a Brownian increment of the drone's command, across it, to
        the command whole, leaving the spin unchanged.

        Args:
            increment: The increment of the command across it, in meters
                       per second.
            observation: What the drone senses.

        Returns:
            The increment of the drone's command, in meters per second, and
            of its spin, in radians per second.
        """
        return increment, zeros_like(observation.spin)

    def steered(self, force: Array, observation: Observation) -> tuple[Array, Array]:
        """
        Adds `force` to the pull, keeping the part of the sum across the
        drone's command where the slot steers and otherwise the part of
        `force` alone, holding the spin at zero.

        Args:
            force: The other slots' acceleration on the drone, in meters per
                   second squared.
            observation: What the drone senses, its weights set.

        Returns:
            The rate of the drone's command, in meters per second squared,
            and of its spin, in radians per second squared.
        """
        pull    = self(observation)
        command = observation.velocity
        return (
            across(pull + force, command) if self.steering
            else pull + across(force, command),
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
            The amplitude on the command, in meters per second per root
            second, and on the force, in meters per second squared per root
            second.
        """
        raise NotImplementedError

    def excited(
        self,
        delay_s     : float,
        draw        : Array,
        observation : Observation,
        step_s      : float,
        turn        : Array
    ) -> Array:
        """
        Computes the state one drone holds after one step of `step_s`,
        leaving it as it stands for a slot whose state no step updates.

        Args:
            delay_s: The age of a broadcast as it arrives, in seconds.
            draw: A number drawn uniformly between zero and one.
            observation: What the drone senses at the step's start, holding
                         the state the step leaves it.
            step_s: The step of the integrator, in seconds.
            turn: The roost's acceleration on the drone, in meters per
                  second squared.
        """
        return observation.force

    def relaxation(self, observation: Observation) -> Array:
        """
        Holds the force on one drone, in meters per second cubed.
        """
        return zeros_like(observation.force)


@documented
class Speed(Component):
    """
    The slot setting the rate of a drone's command along it, returning it in
    meters per second squared.
    """

    fixed: bool = field(default=False, static=True)
    """
    Whether a flight holds each drone at the cruise speed, leaving no speed
    to fluctuate.
    """

    def kicked(self, increment: Array, observation: Observation) -> Array:
        """
        Scales the part of a unit Brownian increment along the drone's
        command into the increment of its speed, in meters per second, none
        for a slot driving no noise of its own.
        """
        return zeros_like(increment)


class Composite(Module):
    """
    The slots the settings nest by name, summed for every drone into the
    rate of its command, beside the radio carrying each drone's state to the
    others and the plant carrying each drone's velocity toward its command.
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
    The random increments on each drone's command and the zig each drone
    starts at the rim or takes up from its neighbors.
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
    The acceleration holding a drone over the roost.
    """

    speed: Speed
    """
    The acceleration along a drone's command.
    """

    def __call__(self, observation: Observation) -> tuple[Array, Array]:
        """
        Turns the cohesion, the contact, the noise, and the roost through
        the heading, which reads the weights the coupling sets, then adds
        the speed slot's acceleration, or keeps only the part across the
        drone's command where the speed slot holds the speed fixed.

        Returns:
            The rate of the drone's command, in meters per second squared,
            and the rate of change of its spin, in radians per second
            squared.
        """
        weighted = tree_at(
            lambda sensed: sensed.weights,
            observation,
            self.coupling(observation)
        )
        turned, spun = self.heading.steered(
            self.cohesion(observation)
            + self.contact(observation)
            + self.noise(observation)
            + self.roost(observation),
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
        a position or a velocity, as the operator a diffrax `ControlTerm`
        reads.

        The noise slot scales the increments of the force and the part of
        each command increment across the command, which the heading splits
        between the command and the spin, and the speed slot scales the part
        along it.
        """
        command, force = vmap(self.noise.amplitudes)(observation)

        def kicked(increment: Flock) -> Flock:
            """
            Scales and splits one set of increments, one per array of the
            flock.
            """
            commands, spins = vmap(self.heading.kicked)(
                command[:, newaxis] * across(increment.commands, observation.velocity),
                observation
            )
            return Flock(
                commands = commands
                + vmap(self.speed.kicked)(increment.commands, observation),
                forces     = force[:, newaxis] * increment.forces,
                positions  = zeros_like(increment.positions),
                spins      = spins,
                velocities = zeros_like(increment.velocities)
            )

        return FunctionLinearOperator(kicked, flock.shapes)

    def drift(self, _: Array, flock: Flock, observation: Observation) -> Flock:
        """
        Computes the rate of change of every drone's state, the vector field
        of a diffrax `ODETerm`, each drone moving over the ground at its
        velocity through the air plus the wind at its position.
        """
        commands, spun = vmap(self)(observation)
        return Flock(
            commands   = commands,
            forces     = vmap(self.noise.relaxation)(observation),
            positions  = flock.velocities + observation.wind,
            spins      = spun,
            velocities = self.plant.tracked(flock.commands, flock.velocities)
        )

    def excited(
        self,
        flock       : Flock,
        key         : Array,
        observation : Observation,
        step_s      : float
    ) -> Flock:
        """
        Advances the state the noise slot holds on each drone by one step
        of `step_s` from the state `flock` holds after the step, each drone
        reading one uniform draw from `key`, the roost's acceleration on it,
        and the radio's `delay_s`.
        """
        return tree_at(
            lambda held: held.forces,
            flock,
            vmap(
                partial(self.noise.excited, delay_s=self.radio.delay_s, step_s=step_s)
            )(
                draw        = uniform(key, flock.forces.shape[:-1]),
                observation = tree_at(
                    lambda sensed: sensed.force,
                    observation,
                    flock.forces
                ),
                turn = vmap(self.roost)(observation)
            )
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
    return accelerations - along(accelerations, velocities)


def along(accelerations: Array, velocities: Array) -> Array:
    """
    Projects each acceleration onto its drone's velocity.
    """
    direction = unit(velocities)
    return (accelerations * direction).sum(-1, keepdims=True) * direction


def closest(count: int, distances: Array) -> tuple[Array, Array]:
    """
    Finds the `count` smallest entries of each row of `distances`, an
    infinite entry marking a column no rank may take.

    Returns:
        The column of each rank, and whether a finite entry fills it.
    """
    nearer, columns = lax.top_k(-distances, count)
    return columns, isfinite(nearer)


def nearest(count: int, points: Array) -> ndarray:
    """
    Queries each drone's `count` nearest other drones on the host, leaving
    a drone out of its own row even where others sit at its exact position,
    which `KDTree.query` may order ahead of it.

    Args:
        count: The number of neighbors per drone.
        points: The positions, `(..., n_drones, 3)`.

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


def zorder(points: Array) -> Array:
    """
    Orders the drones along the Z-order curve through `points`, each axis
    cut into 1,024 steps across the drones' extent, so each drone's code
    takes 30 bits within an `int32` and drones near one another mostly sit
    near one another in the order.
    """
    low   = points.min(0)
    span  = points.max(0) - low
    steps = ((points - low) * 1023 / where(span > 0, span, 1)).astype("int32")
    bits  = arange(10)
    return (
        ((steps[..., newaxis] >> bits & 1) << 3 * bits + arange(3)[:, newaxis])
        .sum((-2, -1))
        .argsort()
    )
