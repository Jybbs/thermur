"""
Holds `Flight`, a `lax.scan` over diffrax's `Euler().step` flying one
flock per seed, beside each statistic its frames give, which the `flight`,
`frames`, `statistics`, and `sweep` steps and `thermur fly` read.
"""

from collections.abc import Callable
from diffrax         import AbstractPath, ControlTerm, Euler, MultiTerm, ODETerm, VirtualBrownianTree
from equinox         import Module, field, tree_at
from itertools       import combinations
from jax             import Array, jit, lax, tree, vmap
from jax.numpy       import cross, where
from jax.random      import fold_in, key as seeded, split
from numpy           import (
    angle,
    arange,
    arccos,
    argmax,
    argsort,
    array,
    array_split,
    asarray,
    bincount,
    clip,
    diff,
    divide,
    einsum,
    exp,
    fft,
    finfo,
    flatnonzero,
    full,
    gradient,
    histogram,
    inf,
    interp,
    isfinite,
    isin,
    linalg,
    linspace,
    log,
    nan,
    ndarray,
    newaxis,
    ones,
    pi,
    sign,
    sort,
    sqrt,
    stack,
    take_along_axis,
    trapezoid,
    unique,
    zeros
)
from scipy.ndimage          import uniform_filter1d
from scipy.signal           import find_peaks, peak_widths
from scipy.sparse.csgraph   import connected_components
from scipy.spatial          import ConvexHull, Delaunay, KDTree
from scipy.spatial.distance import pdist
from scipy.stats            import linregress
from typing                 import Self
from xarray                 import Dataset

from thermur.controller.schemas import Composite, Flock, Inbox, Observation, across, nearest, unit
from thermur.environment.air    import Air
from thermur.metrics.schemas    import Plan


class Flight(Module):
    """
    The kept frames of one flock per seed, `(n_seeds, n_frames, n_drones,
    3)`, beside the composite and the plan that flew them, and under a plan
    setting a turn, the frames of the same flocks flown with no turn.

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

    unturned: Flock | None = None
    """
    The flock at each kept frame flown from the same launch, Brownian path,
    and radio draws with the first drone left unturned, or `None` under a
    plan setting no turn.
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
                integrated(curve, self.plan.frame_s)
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
        Divides the density of the drones on the border of the flock's
        α-shape at `reach_m` by that of the drones inside it, each from the
        cube of its nearest-neighbor distance, which holds a border drone's
        reading free of the empty space beyond the border, over the last
        frames.
        """
        return asarray(
            [
                [
                    ratio_on_border(cloud, self.reach_m, spacing)
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
    def connected(self) -> ndarray:
        """
        Divides the largest group the radio graph of the last frame splits
        into, two drones joined where they sit within the radio's `range_m`,
        by the flock's size.
        """
        return asarray(
            [
                bincount(grouped(cloud, self.composite.radio.range_m)).max()
                / len(cloud)
                for cloud in asarray(self.frames.positions)[:, -1]
            ]
        )

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
        Counts the groups the last frame splits into.
        """
        return asarray([labels.max() + 1 for labels in self.groups], dtype=float)

    @property
    def group_aspect_ratios(self) -> ndarray:
        """
        Averages the shape of the largest group each frame splits into over
        the frames `Plan.shape_frames` picks, `(n_seeds, 3)`, as the length
        of each axis over the thinnest's.
        """
        return asarray(
            [
                [
                    self.largest(frame, seed).aspect_ratios
                    for frame in self.plan.shape_frames
                ]
                for seed in range(self.plan.seeds)
            ]
        ).mean(1)

    @property
    def group_speed_m_s(self) -> ndarray:
        """
        Averages the flock's mean speed through the air over the settled
        frames, in meters per second.
        """
        return linalg.norm(asarray(self.settled.velocities), axis=-1).mean((1, 2))

    @property
    def groups(self) -> list[ndarray]:
        """
        Labels the group each drone of the last frame flies in, one row
        per seed.
        """
        return [
            grouped(cloud, self.reach_m)
            for cloud in asarray(self.frames.positions)[:, -1]
        ]

    @property
    def last(self) -> Flock:
        """
        Keeps the last ten frames, which the neighbor statistics read.
        """
        return tree.map(lambda array: array[:, -10:], self.frames)

    @property
    def middle_aspect_ratio(self) -> ndarray:
        """
        Averages the ratio of the flock's middle axis to its thinnest over
        the settled frames.
        """
        return asarray(self.settled.aspect_ratios)[..., 1].mean(1)

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
        indices   = nearest(self.plan.neighbors, positions)[..., newaxis]
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
    def reach_m(self) -> float:
        """
        Triples `spacing_m`, the hop under which two drones fly in one group
        and the scale of the α-shape whose border `border_density` reads,
        so a gap narrower than the hop joining two drones is no gap in the
        flock, in meters.
        """
        return 3 * self.plan.spacing_m

    @property
    def rearrangement_s(self) -> ndarray:
        """
        Integrates the share of each drone's neighbors at the first settled
        frame still among its neighbors, above the share a random reshuffle
        leaves and up to its first zero, the time over which the neighbors a
        drone reads rearrange, in seconds.
        """
        count  = self.plan.neighbors
        floor  = count / (self.plan.drones - 1)
        stayed = asarray(
            [
                (
                    found[..., newaxis] == found[0, :, newaxis, :]
                ).any(-2).mean((-2, -1))
                for found in nearest(count, asarray(self.settled.positions))
            ]
        )
        return asarray(
            [
                integrated(curve, self.plan.frame_s)
                for curve in (stayed - floor) / (1 - floor)
            ]
        )

    @property
    def settled(self) -> Flock:
        """
        Leaves out the frames before `settled_s`.
        """
        first = self.plan.framed(self.plan.settled_s)
        return tree.map(lambda array: array[:, first:], self.frames)

    @property
    def shares(self) -> ndarray:
        """
        Lays the edges of the distances the speed correlation bins out, as
        shares of the flock's extent.
        """
        return linspace(0.0, 1.0, self.bins.size)

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
        resolves, after removing each mode's time mean, so the peak is the
        lowest frequency the flight resolves where the fluctuations relax
        with no wave.
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
        amplitudes = linalg.norm(
            fft.fft(modes - modes.mean(1, keepdims=True), axis=1),
            axis = -1
        )
        return abs(
            fft.fftfreq(amplitudes.shape[1], self.plan.frame_s)[argmax(
                amplitudes,
                axis = 1
            )]
        )

    @property
    def speed_correlation(self) -> Dataset:
        """
        Reads the correlation $C_{sp}(r)$ of the drones' speed fluctuations
        through the air at each frame `Plan.shape_frames` picks, through
        `correlated`, over the bins `shares` lays out across the extent.

        Returns:
            By seed, the curve averaged over the frames by the share of the
            extent at each bin's center, and over the frames the extent,
            the largest distance between two drones, in meters, and the
            first zero of the curve over the extent, `nan` for a seed whose
            curve stays positive throughout.
        """
        picked = self.plan.shape_frames
        read   = asarray(
            [
                [
                    correlated(cloud, self.shares, linalg.norm(flying, axis=-1))
                    for cloud, flying in zip(clouds, moving, strict=True)
                ]
                for clouds, moving in zip(
                    asarray(self.frames.positions)[:, picked],
                    asarray(self.frames.velocities)[:, picked],
                    strict = True
                )
            ],
            dtype = object
        )
        extents = read[..., 1].astype(float)
        return Dataset(
            {
                "extent_m"                : (("seed", "frame"), extents),
                "speed_correlation"       : (
                    ("seed", "frame", "share"),
                    stack(read[..., 0].tolist())
                ),
                "speed_correlation_ratio" : (
                    ("seed", "frame"),
                    read[..., 2].astype(float) / extents
                )
            },
            coords = {"share": (self.shares[1:] + self.shares[:-1]) / 2}
        ).mean("frame")

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
    def still_statistics(self) -> Dataset:
        """
        Lays out the statistics a flight through still air is set beside the
        starling and drone norms with, by seed, beside the speed correlation
        by the share of the flock's extent.
        """
        return self.speed_correlation.assign(
            {
                name: ("seed", getattr(self, name))
                for name in (
                    "aspect_ratio", "autocorrelation_s", "border_density", "connected",
                    "group_speed_m_s", "middle_aspect_ratio", "polarization",
                    "rearrangement_s", "spacing_m", "wave_speed_m_s"
                )
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
        Reads how the first drone's turn reaches the flock, averaging
        the rotation of each drone's heading away from its heading in the
        unturned flight over shells of equal count at growing distance from
        the first drone, which leaves out each drone's own noise and every
        rotation the turn did not cause.

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

        start            = plan.framed(plan.turn_at_s)
        turned, unturned = (
            asarray(flock.headings)[:, start - 1 :, :, :2] @ array([1, 1j])
            for flock in (self.frames, self.unturned)
        )
        return tuple(
            asarray(
                [
                    response(
                        plan      = plan,
                        distances = linalg.norm(cloud - cloud[0], axis=-1),
                        rotation  = rotation
                    )
                    for rotation, cloud in zip(
                        turned / unturned,
                        asarray(self.frames.positions)[:, start - 1],
                        strict = True
                    )
                ]
            ).T
        )

    @property
    def wave_speed_m_s(self) -> ndarray:
        """
        Fits the speed at which the flock's fastest spontaneous turn over
        the settled frames crosses it, through `crossed`, `nan` for a seed
        whose drones all turn at once, in meters per second.
        """
        plan = self.plan
        return asarray(
            [
                crossed(
                    clouds     = clouds,
                    frame_s    = plan.frame_s,
                    reach_m    = self.reach_m,
                    velocities = flying,
                    window     = plan.window
                )
                for clouds, flying in zip(
                    asarray(self.settled.positions),
                    asarray(self.settled.velocities),
                    strict = True
                )
            ]
        )

    def drifts(self, air: Air, composite: Composite) -> Array:
        """
        Replays each frame but the last through `composite` at the frame's
        own time, every drone hearing the frame each other drone sent
        `delay_s` earlier from within the radio's `range_m` of it, with no
        packet lost, keeping each command's rate across the command, so the
        speed a flight holds enters no residual.

        Returns:
            The drift, `(n_seeds, n_frames - 1, n_drones, 3)`, in meters per
            second squared.
        """
        times = self.plan.times[:-1]
        late  = arange(times.size) - self.plan.framed(composite.radio.delay_s)

        def drift(earlier: Flock, flock: Flock, sent: Flock, t: Array) -> Array:
            """
            Computes the drift across each command of one flock at `t`.
            """
            sensed = Observation.sensed(
                air    = air,
                count  = self.plan.neighbors,
                cruise = self.plan.cruise_m_s,
                flock  = flock,
                inbox  = Inbox.from_broadcast(earlier, composite.radio.range_m, sent),
                t      = t
            )
            return across(vmap(composite)(sensed)[0], flock.commands)

        def taken(indices: ndarray) -> Flock:
            """
            Reads the frames at `indices`, the first frame standing for any
            before it.
            """
            return tree.map(
                lambda leaf: asarray(leaf)[:, clip(indices, 0, None)],
                self.frames
            )

        return jit(vmap(vmap(drift), in_axes=(0, 0, 0, None)))(
            taken(late - self.plan.framed(composite.heading.lag_s)),
            taken(arange(times.size)),
            taken(late),
            times
        )

    @classmethod
    def flown(cls, air: Air, composite: Composite, plan: Plan) -> Self:
        """
        Flies one flock per seed under `vmap`, compiling once for the batch,
        with each launch and each Brownian path deriving from `plan.seed`,
        and under a plan setting a turn, each flock again in the same batch
        with the first drone left unturned.
        """
        lanes = tree.map(
            asarray,
            jit(
                vmap(
                    vmap(
                        lambda key, turning: kept(
                            air       = air,
                            composite = composite,
                            key       = key,
                            plan      = plan,
                            turning   = turning
                        ),
                        in_axes = (0, None)
                    ),
                    in_axes = (None, 0)
                )
            )(
                split(seeded(plan.seed), plan.seeds),
                array([True, False]) if plan.turn_deg else array([True])
            )
        )
        return cls(
            composite = composite,
            frames    = tree.map(lambda leaf: leaf[0], lanes),
            plan      = plan,
            unturned  = tree.map(lambda leaf: leaf[1], lanes) if plan.turn_deg else None
        )

    def largest(self, frame: int, seed: int) -> Flock:
        """
        Keeps the largest group one seed's frame splits into.
        """
        labels = grouped(asarray(self.frames.positions)[seed, frame], self.reach_m)
        return tree.map(
            lambda frames: frames[seed, frame, labels == bincount(labels).argmax()],
            self.frames
        )

    def likelihoods(self, air: Air, composites: dict[str, Composite]) -> Dataset:
        """
        Ranks `composites` by how well each one's drift explains the frames'
        one-step increments, each frame one integrator step apart, through
        a Gaussian likelihood per step fitted to each composite's residuals
        in the plane across each velocity, the form heading likelihoods
        take, and through the coefficient of each drift in a least-squares
        regression of the increments on all of them.

        Returns:
            By composite, `log_likelihood`, per drone and per step, and
            `coefficient`.
        """
        commands = asarray(self.frames.commands)
        rates    = asarray(
            across(diff(commands, axis=1) / self.plan.frame_s, commands[:, :-1])
        )
        drifts = {name: asarray(
            self.drifts(air, item)
        ) for name, item in composites.items()}
        variances = [((rates - drift) ** 2).sum(
            -1
        ).mean() / 2 for drift in drifts.values()]
        library = stack([drift.ravel() for drift in drifts.values()], axis=1)
        return Dataset(
            {
                "coefficient"    : (
                    "model",
                    linalg.lstsq(library, rates.ravel(), rcond=None)[0]
                ),
                "log_likelihood" : (
                    "model",
                    [-(log(2 * pi * value) + 1) for value in variances]
                )
            },
            coords = {"model": list(drifts)}
        )


def alpha_shape(cloud: ndarray, reach_m: float) -> tuple[ndarray, ndarray, float]:
    """
    Builds the α-shape of `cloud` at `reach_m`, the tetrahedra of its
    `Delaunay` triangulation whose circumscribed sphere is narrower than
    `reach_m`, a flat tetrahedron's sphere counting as unbounded.

    Returns:
        Whether each drone sits on a triangle exactly one kept tetrahedron
        holds, the shape's border, whether each sits inside the shape, a
        corner of a kept tetrahedron off the border, so a drone no kept
        tetrahedron reaches counts as neither, and the shape's volume, in
        cubic meters.
    """
    simplices = Delaunay(cloud).simplices
    corners   = cloud[simplices]
    edges     = corners[:, 1:] - corners[:, :1]
    volumes   = abs(linalg.det(edges)) / 6
    solid     = volumes > 0
    squared   = (corners[solid] ** 2).sum(-1)
    centers   = linalg.solve(
        2 * edges[solid],
        (squared[:, 1:] - squared[:, :1])[..., newaxis]
    )[..., 0]
    radii        = full(len(simplices), inf)
    radii[solid] = linalg.norm(centers - corners[solid, 0], axis=-1)
    held         = radii < reach_m
    faces        = sort(
        simplices[held][:, list(combinations(range(4), 3))].reshape(-1, 3),
        axis = 1
    )
    found, counts = unique(faces, axis=0, return_counts=True)
    border        = isin(arange(len(cloud)), found[counts == 1])
    return (
        border,
        isin(arange(len(cloud)), simplices[held]) & ~border,
        float(volumes[held].sum())
    )


def correlated(
    cloud  : ndarray,
    shares : ndarray,
    speeds : ndarray
) -> tuple[ndarray, float, float]:
    """
    Reads the correlation of one frame's speed fluctuations by distance,
    each drone's speed less the flock's mean, summed over the pairs in each
    bin through a weighted `KDTree.count_neighbors` and divided by the pairs
    the bin holds and by the fluctuations' variance.

    Args:
        cloud: The drones' positions, in meters.
        shares: The edges of the bins, as shares of the flock's extent.
        speeds: The drones' speeds through the air, in meters per second.

    Returns:
        The correlation in each bin, `nan` for a bin holding no pair, the
        extent, the largest distance between two drones, and the distance
        where the correlation first falls to zero, interpolated between the
        two bins about it and `nan` where it stays positive, both in meters.
    """
    extent = float(pdist(cloud[ConvexHull(cloud).vertices]).max())
    tree   = KDTree(cloud)
    swings = speeds - speeds.mean()
    edges  = shares * extent
    summed = tree.count_neighbors(
        tree,
        edges,
        cumulative = False,
        weights    = (swings, swings)
    )
    pairs = tree.count_neighbors(tree, edges, cumulative=False)
    curve = divide(
        summed[1:],
        pairs[1:] * (swings**2).mean(),
        out   = full(len(edges) - 1, nan),
        where = pairs[1:] > 0
    )
    held    = isfinite(curve)
    centers = ((edges[1:] + edges[:-1]) / 2)[held]
    found   = curve[held]
    index   = first_zero(found)
    zero    = (
        nan if index == found.size
        else centers[0] if index == 0
        else interp(0.0, found[[index, index - 1]], centers[[index, index - 1]])
    )
    return curve, extent, float(zero)


def crossed(
    clouds     : ndarray,
    frame_s    : float,
    reach_m    : float,
    velocities : ndarray,
    window     : int
) -> float:
    """
    Fits the speed at which one flock's fastest turn crosses it, from the
    radial acceleration of each drone over the turn, the frames where the
    rotation of the flock's mean heading over `window` frames stays above
    half its peak, widened by `window` frames on each side, or the peak's
    frame alone where the peak falls on the first or the last.

    Each pair's delay is the lag maximizing the normalized cross-correlation
    of their radial accelerations, read for one drone against every other at
    once from the product of their spectra, zero-padded to twice the frames
    so no lag wraps around. A drone ranks by how many drones it leads less
    how many lead it, its turning time is the mean, over each drone ranked
    ahead, of that drone's time plus the pair's delay, and the distance
    the turn has crossed by then is the cube root of the drones turned over
    the flock's density inside its α-shape at `reach_m`, fitted against the
    times of the first half of the ranks.

    Args:
        clouds: The drones' positions at each frame, in meters, `(n_frames,
                n_drones, 3)`.
        frame_s: The time between two frames, in seconds.
        reach_m: The scale of the α-shape the density is read inside.
        velocities: The drones' velocities through the air at each frame, in
                    meters per second.
        window: The frames the rotation of the mean heading is read over.

    Returns:
        The speed of the turn, in meters per second, `nan` where every
        drone's turning time is the same.
    """
    headings = asarray(unit(velocities.mean(1)))
    rotation = arccos(
        clip(einsum("ti,ti->t", headings[:-window], headings[window:]), -1, 1)
    )
    peak       = int(argmax(rotation))
    start, end = (
        peak_widths(rotation, [peak])[2:] if 0 < peak < rotation.size - 1
        else ([peak], [peak])
    )
    turning = velocities[max(int(start[0]) - window, 0) : int(end[0]) + 2 * window]
    pushed  = gradient(turning, frame_s, axis=0)
    along   = einsum("tni,tni->tn", pushed, asarray(unit(turning)))
    radial  = sqrt(clip((pushed**2).sum(-1) - along**2, 0, None)).T
    radial  = radial / clip(
        linalg.norm(radial, axis=1, keepdims=True),
        finfo(float).tiny,
        None
    )
    padded  = 2 * radial.shape[1]
    spectra = fft.rfft(radial, padded, axis=1)
    lags    = fft.fftfreq(padded, 1 / padded) * frame_s
    delays  = stack(
        [
            lags[argmax(fft.irfft(spectra * own.conj(), padded, axis=1), axis=1)]
            for own in spectra
        ]
    )
    order = argsort(-sign(delays).sum(1), kind="stable")
    times = zeros(len(order))
    for rank, drone in enumerate(order[1:], start=1):
        ahead        = order[:rank]
        times[drone] = (times[ahead] + delays[ahead, drone]).mean()

    border, inside, volume = alpha_shape(clouds[peak + window // 2], reach_m)
    turned                 = sort(times)[: len(times) // 2]
    if turned.std() == 0 or volume == 0:
        return nan

    reached = (
        arange(1, turned.size + 1) * volume / (border | inside).sum()
    ) ** (1 / 3)
    return float(linregress(turned, reached).slope)


def first_zero(curve: ndarray) -> int:
    """
    Finds the index of the first value at or below zero, or the length of a
    curve holding none.
    """
    crossed = flatnonzero(curve <= 0)
    return crossed[0] if crossed.size else curve.size


def grouped(cloud: ndarray, hop_m: float) -> ndarray:
    """
    Labels the group each drone of `cloud` flies in, two drones sharing a
    group where a chain of hops under `hop_m` joins them.
    """
    return connected_components(
        (tree := KDTree(cloud)).sparse_distance_matrix(tree, hop_m),
        directed = False
    )[1]


def integrated(curve: ndarray, step: float) -> float:
    """
    Integrates `curve` sampled every `step` up to its first zero, the
    estimator an autocorrelation time takes before the noise past that zero
    dominates the sum.
    """
    return trapezoid(curve[: first_zero(curve)], dx=step)


def kept(
    air       : Air,
    composite : Composite,
    key       : Array,
    plan      : Plan,
    turning   : Array | bool = True
) -> Flock:
    """
    Flies one flock, an outer `lax.scan` keeping every `frame_every`-th
    step of an inner one, each inner step hearing over the radio, moving the
    flock through `stepped`, and then setting the first drone's turn where
    `turning` holds.

    Args:
        air: The channels each drone reads.
        composite: The slots the flock flies under.
        key: The key the launch, the Brownian path, each lost packet, and
             each step of the noise slot's state derive from.
        plan: The plan the flock flies to.
        turning: Whether the first drone turns, which leaves every draw as
                 it is either way.

    Returns:
        The flock at each kept frame.
    """
    launch_key, path_key, radio_key, noise_key = split(key, 4)
    launch = plan.launched(launch_key)
    path   = VirtualBrownianTree(
        0.0,
        plan.duration_s + plan.step_s,
        key   = path_key,
        shape = launch.shapes,
        tol   = plan.resolution_s
    )
    radio = composite.radio
    lag   = round(composite.heading.lag_s / plan.step_s)

    def advance(
        state : tuple[Flock, Inbox, Flock],
        step  : Array
    ) -> tuple[tuple[Flock, Inbox, Flock], None]:
        """
        Advances the flock, what each drone heard, and the broadcasts the
        radio keeps by one step.
        """
        flock, inbox, sent = state
        t           = step * plan.step_s
        heard, sent = radio.heard(
            flock,
            inbox,
            radio_key,
            lag,
            sent,
            step,
            plan.step_s
        )
        moved = stepped(
            air       = air,
            composite = composite,
            flock     = flock,
            inbox     = heard,
            key       = fold_in(noise_key, step),
            path      = path,
            plan      = plan,
            t         = t
        )
        turned = tree_at(
            lambda held: held.commands,
            moved,
            moved.commands.at[0].set(
                where((t >= plan.turn_at_s) & turning, plan.turned, moved.commands[0])
            )
        ) if plan.turn_deg else moved
        return (turned, heard, sent), None

    def frame(
        state : tuple[Flock, Inbox, Flock],
        start : Array,
        /
    ) -> tuple[tuple[Flock, Inbox, Flock], Flock]:
        """
        Advances the flock through `frame_every` steps, keeping the last.
        """
        ahead = lax.scan(advance, state, start + lax.iota("int32", plan.frame_every))[0]
        return ahead, ahead[0]

    return lax.scan(
        frame,
        (launch, *radio.opened(launch, lag, plan.step_s)),
        plan.frame_every * lax.iota("int32", plan.frames)
    )[1]


def ratio_on_border(cloud: ndarray, reach_m: float, spacings: ndarray) -> float:
    """
    Divides the mean density of the drones on the border of the α-shape of
    `cloud` at `reach_m` by that of the drones inside it, each from the cube
    of its distance in `spacings` to its nearest neighbor, `nan` where no
    drone sits inside it.
    """
    border, inside, _ = alpha_shape(cloud, reach_m)
    density           = spacings**-3.0
    return (
        density[border].mean() / density[inside].mean()
        if inside.any() else nan
    )


def response(
    distances : ndarray,
    plan      : Plan,
    rotation  : ndarray
) -> tuple[float, float]:
    """
    Fits how one flock's shells of drones respond to a turn.

    Args:
        distances: Each drone's distance from the first drone as the turn
                   began, in meters.
        plan: The plan the flock flew to.
        rotation: The rotation of each drone's heading away from its heading
                  in the unturned flight, from the frame before the turn
                  began, as a complex number on the unit circle, `(n_frames,
                  n_drones)`.

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


def stepped(
    air       : Air,
    composite : Composite,
    flock     : Flock,
    inbox     : Inbox,
    key       : Array,
    path      : AbstractPath | Callable[[Array, Array], Flock],
    plan      : Plan,
    t         : Array
) -> Flock:
    """
    Moves one flock through one step of the composite, sensing once from
    what each drone heard, advancing through one `Euler().step` over the
    drift and the Brownian increments `path` gives between `t` and `t +
    plan.step_s`, then applying the discrete updates:

    - The turn of each command at the change in spin the step reaches
    - The speed, the cruise speed where the speed slot holds it and
      otherwise the stepped command's part along the drone's last command
    - The plant's bounds on each command
    - The step of the noise slot's state

    Args:
        air: The channels each drone reads.
        composite: The slots the flock flies under.
        flock: The flock at `t`.
        inbox: What each drone heard of every other by `t`.
        key: The key the noise slot's draws at this step derive from.
        path: The Brownian path the increments are read from, or a function
              of the step's two ends returning them.
        plan: The plan the flock flies to.
        t: The time, in seconds since the flight started.

    Returns:
        The flock at `t + plan.step_s`.
    """
    sensed = Observation.sensed(
        air    = air,
        count  = plan.neighbors,
        cruise = plan.cruise_m_s,
        flock  = flock,
        inbox  = inbox,
        t      = t
    )
    terms = MultiTerm(ODETerm(composite.drift), ControlTerm(composite.diffusion, path))
    moved, *_ = Euler().step(terms, t, t + plan.step_s, flock, sensed, None, False)
    turned    = tree_at(
        lambda held: held.commands,
        moved,
        moved.commands
        + plan.step_s * cross(moved.spins - flock.spins, flock.commands)
    ).cruising(
        plan.cruise_m_s if composite.speed.fixed
        else (moved.commands * unit(flock.commands)).sum(-1, keepdims=True)
    )
    held = tree_at(
        lambda kept_flock: kept_flock.commands,
        turned,
        composite.plant.held(turned.commands)
    )
    return composite.excited(held, key, sensed, plan.step_s)
