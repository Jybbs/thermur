"""
Holds `Flight`, a `lax.scan` over diffrax's `Euler().step` flying one
flock per seed, beside each statistic its frames give, which the `flight`,
`frames`, and `statistics` steps and `thermur fly` read.
"""

from diffrax    import ControlTerm, Euler, MultiTerm, ODETerm, VirtualBrownianTree
from equinox    import Module, field, tree_at
from jax        import Array, ShapeDtypeStruct, jit, lax, tree, vmap
from jax.numpy  import roll, where
from jax.random import key as seeded, split
from numpy         import (
    angle,
    arange,
    arccos,
    argmax,
    argsort,
    array,
    array_split,
    asarray,
    clip,
    diff,
    einsum,
    exp,
    fft,
    flatnonzero,
    histogram,
    isin,
    linalg,
    linspace,
    log,
    nan,
    ndarray,
    newaxis,
    ones,
    pi,
    stack,
    take_along_axis,
    trapezoid
)
from scipy.ndimage          import uniform_filter1d
from scipy.signal           import find_peaks
from scipy.sparse.csgraph   import connected_components
from scipy.spatial          import ConvexHull, KDTree
from scipy.spatial.distance import pdist
from scipy.stats            import linregress
from typing                 import Self
from xarray                 import Dataset

from thermur.controller.schemas import Composite, Flock, Observation, across, nearest, unit
from thermur.environment.air    import Air
from thermur.metrics.schemas    import Plan


class Flight(Module):
    """
    The kept frames of one flock per seed, `(n_seeds, n_frames, n_drones,
    3)`, beside the composite and the plan that flew them.

    Each statistic reads the frames on the host, one value per seed unless
    it says otherwise.
    """

    composite: Composite
    """
    The slots the flight flew under.
    """

    frames: Flock
    """
    The flock at each kept frame.
    """

    plan: Plan = field(static=True)
    """
    The plan the flight flew to.
    """

    @property
    def alignment_by_distance(self) -> ndarray:
        """
        Averages the dot product of a drone's heading and each neighbor's
        by the neighbor's distance, over the last frames, in the bins of
        `bins`, `(n_seeds, n_bins)`, a bin holding no neighbor reading zero.
        """
        return stack(
            [
                histogram(apart, self.bins, weights=product)[0]
                / clip(histogram(apart, self.bins)[0], 1, None)
                for apart, product in zip(*self.neighbor_pairs, strict=True)
            ]
        )

    @property
    def alignment_by_rank(self) -> ndarray:
        """
        Averages the dot product of a drone's heading and each neighbor's by
        the neighbor's rank, over the last frames, `(n_seeds, n_neighbors)`.
        """
        return self.neighbor_pairs[1].mean((1, 2))

    @property
    def aspect_ratio(self) -> ndarray:
        """
        Averages the ratio of the flock's longest axis to its thinnest over
        the settled frames.
        """
        return asarray(self.settled.aspect_ratios)[..., -1].mean(1)

    @property
    def autocorrelation_s(self) -> ndarray:
        """
        Integrates the autocorrelation of each drone's heading relative to
        the flock's, up to its first zero, in seconds.
        """
        headings = asarray(self.settled.headings)
        relative = headings - headings.mean(2, keepdims=True)
        frames   = relative.shape[1]
        spectrum = fft.rfft(relative, 2 * frames, axis=1)
        lagged   = fft.irfft(abs(spectrum) ** 2, axis=1)[:, :frames].sum(-1).mean(-1)
        return asarray(
            [
                trapezoid(curve[: first_zero(curve)], dx=self.plan.frame_s)
                for curve in lagged / (frames - arange(frames)) / lagged[:, :1] * frames
            ]
        )

    @property
    def axes(self) -> tuple[str, ...]:
        """
        Names the dimensions of a frame's positions or velocities.
        """
        return ("seed", "time", "drone", "axis")

    @property
    def bins(self) -> ndarray:
        """
        Lays the edges of the distances the statistics bin out, to four
        times `spacing_m`, in meters.
        """
        return linspace(0.0, 4 * self.plan.spacing_m, 33)

    @property
    def border_density(self) -> ndarray:
        """
        Divides the density of the drones on the flock's convex hull by that
        of the rest, each from the cube of its nearest-neighbor distance,
        which holds a border drone's reading free of the empty space beyond
        the hull, over the last frames.
        """
        return asarray(
            [
                [
                    ratio_on_hull(cloud, spacing)
                    for cloud, spacing in zip(frames, spacings, strict=True)
                ]
                for frames, spacings in zip(
                    asarray(self.last.positions),
                    self.neighbor_pairs[0][..., 0],
                    strict = True
                )
            ]
        ).mean(1)

    @property
    def centers(self) -> ndarray:
        """
        Finds the center of each bin `bins` lays out, in meters.
        """
        return (self.bins[1:] + self.bins[:-1]) / 2

    @property
    def dataset(self) -> Dataset:
        """
        Lays the kept frames out as the table a recorded run saves, with the
        time of each frame in seconds.
        """
        return Dataset(
            {
                "positions"  : (self.axes, asarray(self.frames.positions)),
                "velocities" : (self.axes, asarray(self.frames.velocities))
            },
            coords = {"time": self.plan.times}
        )

    @property
    def fragments(self) -> ndarray:
        """
        Counts the groups the last frame splits into, two drones sharing
        a group where a chain of hops under three times `spacing_m` joins
        them.
        """
        return asarray(
            [
                connected_components(
                    (tree := KDTree(cloud)).sparse_distance_matrix(
                        tree,
                        3 * self.plan.spacing_m
                    ),
                    directed = False
                )[0]
                for cloud in asarray(self.frames.positions)[:, -1]
            ],
            dtype = float
        )

    @property
    def last(self) -> Flock:
        """
        Keeps the last ten frames, which the neighbor statistics read.
        """
        return tree.map(lambda array: array[:, -10:], self.frames)

    @property
    def neighbor_pairs(self) -> tuple[ndarray, ndarray]:
        """
        Queries each drone's neighbors over the last frames.

        Returns:
            Each neighbor's distance and the dot product of its heading with
            the drone's, each `(n_seeds, n_frames, n_drones, n_neighbors)`.
        """
        positions = asarray(self.last.positions)
        headings  = asarray(self.last.headings)
        indices   = nearest(positions, self.plan.neighbors)[..., newaxis]
        spots     = take_along_axis(positions[..., newaxis, :, :], indices, axis=-2)
        others    = take_along_axis(headings[..., newaxis, :, :], indices, axis=-2)
        return (
            linalg.norm(spots - positions[..., newaxis, :], axis=-1),
            einsum("...d,...rd->...r", headings, others)
        )

    @property
    def peak(self) -> tuple[ndarray, ndarray]:
        """
        Finds the first peak of the pair distribution $g(r)$ over the last
        frames, which divides the histogram of the pair distances by the
        volume of each shell.

        Returns:
            The distance of each seed's peak, in meters, and its height over
            the mean of the distribution, both `nan` for a distribution with no
            peak.
        """
        centers = self.centers
        peaks   = []
        for frames in asarray(self.last.positions):
            pairs = stack([pdist(cloud) for cloud in frames]).ravel()
            shape = histogram(pairs, self.bins)[0] / centers**2
            found = find_peaks(shape)[0]
            peaks.append(
                (centers[found[0]], shape[found[0]] / shape.mean())
                if found.size else (nan, nan)
            )

        return tuple(asarray(peaks).T)

    @property
    def polarization(self) -> ndarray:
        """
        Averages the flock's polarization over the settled frames.
        """
        return asarray(self.settled.polarization).mean(1)

    @property
    def settled(self) -> Flock:
        """
        Leaves out the frames before `settled_s`.
        """
        first = self.plan.framed(self.plan.settled_s)
        return tree.map(lambda array: array[:, first:], self.frames)

    @property
    def spacing_m(self) -> ndarray:
        """
        Averages each drone's distance to its nearest neighbor over the last
        frames, in meters.
        """
        return self.neighbor_pairs[0][..., 0].mean((1, 2))

    @property
    def spectral_peak_hz(self) -> ndarray:
        """
        Finds the frequency at which the spontaneous correlation
        $C(k,\\omega)$ of the velocity fluctuations peaks, at the smallest
        wave vector along the flock's mean heading the flock's own length
        resolves, zero where the fluctuations relax with no wave.
        """
        flock      = self.settled
        velocities = asarray(flock.velocities)
        positions  = asarray(flock.positions)
        along      = einsum(
            "sfnd,sd->sfn",
            positions - positions.mean(2, keepdims=True),
            asarray(unit(velocities.mean((1, 2))))
        )
        modes = (
            (velocities - velocities.mean(2, keepdims=True))
            * exp(
                2j * pi * along / (4 * along.std((1, 2), keepdims=True))
            )[..., newaxis]
        ).mean(2)
        rates = fft.fftfreq(modes.shape[1], self.plan.frame_s)
        return abs(rates[argmax((abs(fft.fft(modes, axis=1)) ** 2).sum(-1), axis=1)])

    @property
    def statistics(self) -> Dataset:
        """
        Lays out every statistic the frames give as the table a recorded
        run saves, by seed, each distance bin at its center in multiples of
        the launch's `spacing_m`, so flights launched at different spacings
        share their bins.
        """
        return Dataset(
            {
                "alignment_by_distance" : (
                    ("seed", "distance"),
                    self.alignment_by_distance
                ),
                "alignment_by_rank"     : (("seed", "rank"), self.alignment_by_rank),
                **{
                    name: ("seed", getattr(self, name))
                    for name in (
                        "aspect_ratio", "autocorrelation_s", "border_density",
                        "fragments", "polarization", "spacing_m",
                        "spectral_peak_hz", "turn_rate_per_s"
                    )
                },
                **{
                    name: ("seed", value)
                    for name, value in zip(
                        ("peak_m", "peak_height", "turn_speed_m_s", "turn_reach"),
                        (*self.peak, *self.turn_response),
                        strict = True
                    )
                }
            },
            coords = {
                "distance" : self.centers / self.plan.spacing_m,
                "rank"     : arange(1, self.plan.neighbors + 1)
            }
        )

    @property
    def turn_rate_per_s(self) -> ndarray:
        """
        Counts the rate at which the flock's mean heading starts rotating
        faster than `turning_deg_s`.
        """
        plan   = self.plan
        window = plan.window
        mean   = asarray(unit(asarray(self.settled.headings).mean(2)))
        cosine = clip(einsum("sfd,sfd->sf", mean[:, window:], mean[:, :-window]), -1, 1)
        fast   = arccos(cosine) / plan.turning_window_s > plan.turning_rad_s
        starts = (diff(fast.astype(int), axis=1) == 1).sum(1)
        return starts / (fast.shape[1] * plan.frame_s)

    @property
    def turn_response(self) -> tuple[ndarray, ndarray]:
        """
        Reads how the first drone's turn reaches the flock, averaging the
        rotation of each drone's heading since the turn began over shells of
        equal count at growing distance from the first drone, which leaves
        out each drone's own noise.

        Returns:
            The speed of the response across the flock, from a line through
            each shell's distance against the first frame it has rotated
            through half the turn, in meters per second, and the farthest
            shell's rotation at the flight's end over the turn. Both are `nan`
            for a plan with no turn.
        """
        plan = self.plan
        if not plan.turn_deg:
            return (nan * ones(plan.seeds),) * 2

        start = plan.framed(plan.turn_at_s)
        since = asarray(self.frames.headings)[:, start - 1 :, :, :2] @ array([1, 1j])
        return tuple(
            asarray(
                [
                    response(
                        plan      = plan,
                        distances = linalg.norm(cloud - cloud[0], axis=-1),
                        rotation  = rotation
                    )
                    for rotation, cloud in zip(
                        since / since[:, :1],
                        asarray(self.frames.positions)[:, start - 1],
                        strict = True
                    )
                ]
            ).T
        )

    def drifts(self, composite: Composite, air: Air) -> Array:  # prose: ignore[unsorted-positionals]
        """
        Replays each frame but the last through `composite` at the frame's
        own time, keeping each acceleration's part across the drone's
        velocity, so the speed a flight holds enters no residual.

        Returns:
            The drift, `(n_seeds, n_frames - 1, n_drones, 3)`, in meters per
            second squared.
        """
        velocities = asarray(self.frames.velocities)
        lag        = self.plan.framed(composite.heading.lag_s)
        times      = self.plan.times[:-1]

        def drift(  # prose: ignore[unsorted-positionals]
            forces    : Array,
            positions : Array,
            past      : Array,
            moving    : Array,
            t         : Array
        ) -> Array:
            """
            Computes the drift across each velocity of one flock at `t`.
            """
            sensed = Observation.sensed(
                Flock(forces=forces, positions=positions, velocities=moving),
                past,
                air,
                t,
                self.plan.neighbors
            )
            return across(vmap(composite)(sensed), moving)

        return jit(
            vmap(vmap(drift), in_axes=(0, 0, 0, 0, None))
        )(
            asarray(self.frames.forces)[:, :-1],
            asarray(self.frames.positions)[:, :-1],
            velocities[:, clip(arange(times.size) - lag, 0, None)],
            velocities[:, :-1],
            times
        )

    @classmethod
    def flown(cls, composite: Composite, air: Air, plan: Plan) -> Self:  # prose: ignore[unsorted-positionals]
        """
        Flies one flock per seed under `vmap`, compiling once for the batch,
        with each launch and each Brownian path deriving from `plan.seed`.
        """
        return cls(
            composite = composite,
            frames    = tree.map(
                asarray,
                jit(
                    vmap(
                        lambda key: kept(
                            air       = air,
                            composite = composite,
                            key       = key,
                            plan      = plan
                        )
                    )
                )(split(seeded(plan.seed), plan.seeds))
            ),
            plan = plan
        )

    def likelihoods(self, composites: dict[str, Composite], air: Air) -> Dataset:  # prose: ignore[unsorted-positionals]
        """
        Ranks `composites` by how well each one's drift explains the frames'
        one-step increments, each frame one integrator step apart, through
        a Gaussian likelihood per step fitted to each composite's residuals,
        the form heading likelihoods take, and through the coefficient of
        each drift in a least-squares regression of the increments on all
        of them.

        Returns:
            By composite, `log_likelihood`, per drone and per step, and
            `coefficient`.
        """
        velocities = asarray(self.frames.velocities)
        rates      = asarray(
            across(diff(velocities, axis=1) / self.plan.frame_s, velocities[:, :-1])
        )
        drifts = {name: asarray(
            self.drifts(item, air)
        ) for name, item in composites.items()}
        variances = [((rates - drift) ** 2).mean() for drift in drifts.values()]
        library   = stack([drift.ravel() for drift in drifts.values()], axis=1)
        return Dataset(
            {
                "coefficient"    : (
                    "model",
                    linalg.lstsq(library, rates.ravel(), rcond=None)[0]
                ),
                "log_likelihood" : (
                    "model",
                    [-1.5 * (log(2 * pi * value) + 1) for value in variances]
                )
            },
            coords = {"model": list(drifts)}
        )


def first_zero(curve: ndarray) -> int:
    """
    Finds the index of the first value at or below zero, or the length of a
    curve holding none.
    """
    crossed = flatnonzero(curve <= 0)
    return crossed[0] if crossed.size else curve.size


def kept(air: Air, composite: Composite, key: Array, plan: Plan) -> Flock:
    """
    Flies one flock, an outer `lax.scan` keeping every `frame_every`-th step
    of an inner one, each inner step sensing once, advancing through one
    `Euler().step`, then applying the discrete updates, meaning the cruise
    speed where the heading holds it and the first drone's turn.

    Args:
        air       : The channels each drone reads.
        composite : The slots the flock flies under.
        key       : The key the launch and the Brownian path derive from.
        plan      : The plan the flock flies to.

    Returns:
        The flock at each kept frame.
    """
    launch_key, path_key = split(key)
    launch               = plan.launched(launch_key)
    path                 = VirtualBrownianTree(
        0.0,
        plan.duration_s + plan.step_s,
        key   = path_key,
        shape = tree.map(lambda leaf: ShapeDtypeStruct(leaf.shape, leaf.dtype), launch),
        tol   = plan.resolution_s
    )
    terms  = MultiTerm(ODETerm(composite.drift), ControlTerm(composite.diffusion, path))
    solver = Euler()
    lag    = max(round(composite.heading.lag_s / plan.step_s), 1)

    def advance(
        state : tuple[Flock, Array],
        step  : Array
    ) -> tuple[tuple[Flock, Array], None]:
        """
        Advances the flock and the velocities it remembers by one step.
        """
        flock, past = state
        t           = step * plan.step_s
        sensed      = Observation.sensed(flock, past[0], air, t, plan.neighbors)
        moved, *_   = solver.step(terms, t, t + plan.step_s, flock, sensed, None, False)
        if composite.heading.confined:
            moved = moved.cruising(plan.cruise_m_s)

        turned = tree_at(
            lambda held: held.velocities,
            moved,
            moved.velocities.at[0].set(
                where(t >= plan.turn_at_s, plan.turned, moved.velocities[0])
            )
        ) if plan.turn_deg else moved
        return (turned, roll(past, -1, axis=0).at[-1].set(moved.velocities)), None

    def frame(  # prose: ignore[unsorted-positionals]
        state : tuple[Flock, Array],
        start : Array
    ) -> tuple[tuple[Flock, Array], Flock]:
        """
        Advances the flock through `frame_every` steps, keeping the last.
        """
        ahead = lax.scan(advance, state, start + lax.iota("int32", plan.frame_every))[0]
        return ahead, ahead[0]

    return lax.scan(
        frame,
        (launch, launch.velocities[None].repeat(lag, axis=0)),
        plan.frame_every * lax.iota("int32", plan.frames)
    )[1]


def ratio_on_hull(cloud: ndarray, spacings: ndarray) -> float:
    """
    Divides the mean density of the drones on the convex hull of `cloud` by
    that of the rest, each from the cube of its distance in `spacings` to
    its nearest neighbor.
    """
    border  = isin(arange(len(cloud)), ConvexHull(cloud).vertices)
    density = spacings**-3.0
    return density[border].mean() / density[~border].mean()


def response(
    distances : ndarray,
    plan      : Plan,
    rotation  : ndarray
) -> tuple[float, float]:
    """
    Fits how one flock's shells of drones respond to a turn.

    Args:
        distances : Each drone's distance from the first drone as the turn
                    began, in meters.
        plan      : The plan the flock flew to.
        rotation  : The rotation of each drone's heading since the
                    turn began, as a complex number on the unit circle,
                    `(n_frames, n_drones)`.

    Returns:
        The speed of the response, in meters per second, zero where fewer than
        three shells respond at distinct times, and the farthest shell's
        rotation at the flight's end over the turn.
    """
    shells = array_split(argsort(distances)[1:], 5)
    curves = abs(
        angle(
            uniform_filter1d(
                stack([rotation[:, shell].mean(1) for shell in shells]),
                plan.window // 5 + 1,
                axis = 1
            )
        )
    )
    turn   = abs(plan.turn_rad)
    joined = curves >= turn / 2
    joins  = joined.any(1)
    times  = plan.frame_s * argmax(joined[joins], axis=1)
    reach  = float(curves[-1, -plan.window :].mean() / turn)
    if joins.sum() < 3 or times.std() == 0:
        return 0.0, reach

    spans = asarray([distances[shell].mean() for shell in shells])[joins]
    return max(float(linregress(times, spans).slope), 0.0), reach
