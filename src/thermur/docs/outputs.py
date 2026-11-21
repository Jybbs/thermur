"""
Holds `Outputs`, the files the `docs:render` task writes into the site
and every Markdown page git tracks, which `repo:audit` holds the committed
copies to, beside `tabled`, which lays a record's fields out as the table a
page's region holds, `specimens`, which lays out the identity as the regions
of The Visual Identity, and `pieces`, which lays out the kit's pieces as the
region of The Kit.
"""

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses     import asdict, fields, replace
from equinox         import tree_at
from functools       import cached_property, partial
from jax             import Array, numpy as jnp, tree, vmap
from jax.random      import key as seeded, uniform
from json            import dumps
from math            import sqrt
from numpy           import (
    arange, around, asarray, cross, geomspace, linalg, ndarray, zeros
)
from numpy.random import default_rng
from pathlib      import Path
from pydantic     import BaseModel, Field, TypeAdapter, create_model

from thermur.controller.schemas import (
    Composite, Flock, Inbox, Observation, Radio, unit, zorder
)
from thermur.docs.kit        import Piece
from thermur.docs.pages      import Page
from thermur.environment.air import Air
from thermur.environment.schemas import (
    BASE_K,
    Crop,
    GRAVITY_M_S2,
    HEAT_CAPACITY_J_KG_K,
    ParametricThreat,
    Ranges,
    Sample
)
from thermur.identity.schemas  import Identity, Pair, Palette
from thermur.metrics.flight    import stepped
from thermur.metrics.schemas   import Plan
from thermur.metrics.trials    import Trial
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout
from thermur.repo.schemas      import Finding


class Outputs(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The files the render task writes under the root of `checkout`, each
    named relative to it.
    """

    checkout: Checkout = Field(default_factory=Checkout)
    """
    The clone the task writes into, the working directory by default.
    """

    composite: Path = Path("docs/.vitepress/lib/rendered/composite.json")
    """
    The file the site's flocks read each default slot's kind and parameters
    from, beside the fields of the plan their launch and their step read.
    """

    crop: Path = Path("docs/.vitepress/lib/rendered/crop.json")
    """
    The file the Data page's sketch of the part kept reads the crop's defaults
    and the sample's length and snapshot times from.
    """

    identity: Path = Path("docs/.vitepress/lib/rendered/identity.json")
    """
    The file the site's theme and its preview cards read the identity from.
    """

    parity: Path = Path("docs/.vitepress/lib/rendered/parity.json")
    """
    The file the site's suite reads three small flocks from, each beside its
    Brownian increments, the uniform draw each drone's zig reads, and the
    flock one step of the package's composite moves it to, a run of the
    package's radio beside the draws each packet's loss reads, a run of it
    losing no packet, and the order the package's `zorder` gives a cloud of
    drones.
    """

    threat: Path = Path("docs/.vitepress/lib/rendered/threat.json")
    """
    The file the Data page's drawing of the front reads the default record and
    the constants its air is worked out from.
    """

    trials: Path = Path("docs/.vitepress/lib/rendered/trials.json")
    """
    The file the charts on The Interaction Rules read what each condition of
    a trial sets from.
    """

    @property
    def drifted(self) -> list[Path]:
        """
        Lists each file whose text on disk differs from what the render
        would write, a missing file among them.
        """
        return [
            path
            for path, text in self.written.items()
            if not (file := self.checkout.root / path).is_file()
            or file.read_text(encoding="utf-8") != text
        ]

    @property
    def refused(self) -> list[Finding]:
        """
        Names each region `regions` declares whose page the checkout lacks
        or whose page does not carry its opening marker and then its closing
        one exactly once each, which the render leaves as it stands, page by
        page in the order `regions` lists them.
        """
        pages = self.checkout.pages
        return [
            Finding(
                file    = path,
                message = f"`{path}` does not carry the opening and then the closing "
                f"marker of its `{name}` region once each, so `mise render` leaves "
                "the region as it stands"
            )
            for path, regions in self.regions.items()
            for name in regions
            if path not in pages or not pages[path].bounds(name)
        ]

    @cached_property
    def regions(self) -> dict[Path, dict[str, str]]:
        """
        Renders the body of each region the render task writes into a page,
        by the page's path and then by the region's name, meaning the fields
        `Sample`, `Crop`, `Ranges`, and `ParametricThreat` declare on the
        Data page, on The Interaction Rules the fields `Plan`, the contact,
        the plant, the radio, and the roost declare beside those of each
        open slot's default and each of its candidates, every one named for
        its component, on The Visual Identity the identity as `specimens`
        lays it out, and on The Kit the pieces each declaration in the clone
        records.
        """
        defaults = Settings()
        return {
            Path("docs/pages/reference/identity.md") : specimens(Identity.shipped()),
            Path("docs/pages/reference/kit.md")      : {
                "pieces": pieces(Piece.declared(self.checkout.root))
            },
            Path("docs/pages/data/index.md")         : {
                "crop"   : tabled(Crop),
                "ranges" : tabled(Ranges),
                "sample" : tabled(Sample),
                "threat" : tabled(ParametricThreat)
            },
            Path("docs/pages/methods/rules.md")      : {
                slot.name: tabled(type(getattr(defaults, slot.name)))
                for slot in fields(Composite)
                if slot.name not in {trial.slot for trial in Trial.listed()}
            } | {"plan": tabled(Plan)} | {
                trial.slot: tabled(
                    kind = create_model(
                        trial.slot,
                        **{
                            type(component).__name__.lower(): (
                                type(component),
                                component
                            )
                            for component in (
                                getattr(defaults, trial.slot),
                                *trial.candidates.values()
                            )
                        }
                    )
                )
                for trial in Trial.listed()
            }
        }

    @property
    def unknown(self) -> list[Finding]:
        """
        Names each marker a page carries under a study's key that the
        bibliography holds no entry for, page by page in path order, at the
        line of the block it sits in.
        """
        bibliography = self.checkout.bibliography
        return [
            Finding(
                file    = path,
                message = f"Line {line} of `{path}` cites `{key}`, which names no "
                f"entry in `{bibliography.file}`"
            )
            for path, page in self.checkout.pages.items()
            for line, key in page.unknown(bibliography)
        ]

    @cached_property
    def written(self) -> dict[Path, str]:
        """
        Renders the text of each file the render task writes, by its path,
        the identity as JSON with its fields in the order the records
        declare them, the crop's heights as JSON beside the sample's length
        and snapshot times, the default threat as JSON beside the constants
        its air is worked out from, the changes each condition of each trial
        makes to the plan as JSON by slot, and each page with its regions
        as `Page.filled` writes them and its definitions as `Page.cited`
        writes them.
        """
        crop   = Crop().model_dump(include={"floor_m", "levels", "spacing_m"})
        sample = Sample()
        threat = {
            "base_k"               : BASE_K,
            "gravity_m_s2"         : GRAVITY_M_S2,
            "heat_capacity_j_kg_k" : HEAT_CAPACITY_J_KG_K
        } | ParametricThreat().model_dump()
        settings  = Settings()
        composite = Composite.from_settings(settings)
        model     = {
            slot.name: {"kind": type(component).__name__} | asdict(component)
            for slot in fields(Composite)
            if (component := getattr(composite, slot.name))
        }
        model["radio"]["slots"] = composite.radio.slots
        return {
            self.composite : dumps(
                model
                | {
                    "plan": settings.plan.model_dump(
                        include = {
                            "axes", "center_m", "clearance_sd",
                            "course_deg", "cruise_m_s", "neighbors",
                            "scatter_deg", "spacing_m", "step_s"
                        }
                    )
                },
                default = float,
                indent  = 2
            ) + "\n",
            self.crop      : dumps(
                crop | {"length_m": sample.length_m, "times_s": sample.times_s},
                indent = 2
            ) + "\n",
            self.identity  : Identity.shipped().model_dump_json(indent=2) + "\n",
            self.parity    : dumps(paired(composite, settings.plan), indent=2) + "\n",
            self.threat    : dumps(threat, indent=2) + "\n",
            self.trials    : dumps(
                {trial.slot: trial.conditions for trial in Trial.listed()},
                indent = 2
            ) + "\n"
        } | {
            path: Page(text=page.filled(self.regions.get(path, {}))).cited(
                self.checkout.bibliography
            )
            for path, page in self.checkout.pages.items()
        }

    def write(self) -> list[Path]:
        """
        Writes each drifted file, leaving every current one untouched.

        Returns:
            The paths of the files it wrote, in the order `written` lists
            them.
        """
        for path in (drifted := self.drifted):
            (file := self.checkout.root / path).parent.mkdir(
                exist_ok = True,
                parents  = True
            )
            file.write_text(self.written[path], encoding="utf-8")

        return drifted


def bounded(field: Mapping[str, object]) -> str:
    """
    Phrases the numeric bounds a field's JSON Schema sets in words, the
    lower before the upper and joined by `and`, such as `above 0` for an
    `exclusiveMinimum` of zero, or as an empty string where the schema sets
    none.
    """
    words = (
        ("exclusiveMinimum", "above"),
        ("minimum", "at least"),
        ("exclusiveMaximum", "below"),
        ("maximum", "at most")
    )
    return " and ".join(f"{word} {field[key]}" for key, word in words if key in field)


def colored(name: str, palette: Palette) -> str:
    """
    Names the color `palette` files under `name` in words beside its hex in
    code, such as ``Fire Lord Deep `#995500` `` for `fire-lord-deep`.
    """
    return f"{name.replace('-', ' ').title()} `{palette.colors[name]}`"


def described(kind: type[BaseModel], name: str) -> str:
    """
    Reads the description of the field `name` that `kind` declares on one
    line, without its closing period.
    """
    return " ".join(str(kind.model_fields[name].description).split()).removesuffix(".")


def fielded(
    definitions : Mapping[str, Mapping[str, object]],
    properties  : Mapping[str, Mapping[str, object]],
    *,
    prefix      : str = ""
) -> Iterator[tuple[str, str, str, str]]:
    """
    Reads the name, the default, the bound, and the description of each
    field among a JSON Schema's `properties`, resolving a reference into
    `definitions`, and reads a field holding a record of its own as that
    record's fields, each named after it with a dot.
    """
    for name, field in properties.items():
        resolved = definitions.get(
            str(field.get("$ref", "")).rpartition("/")[2],
            {}
        ) | field
        if "properties" in resolved:
            yield from fielded(
                definitions = definitions,
                prefix      = f"{prefix}{name}.",
                properties  = resolved["properties"]
            )
        else:
            yield (
                f"{prefix}{name}",
                str(resolved.get("default")),
                bounded(resolved),
                " ".join(str(resolved.get("description", "")).split()).removesuffix(".")
            )


def flown(
    composite : Composite,
    draws     : ndarray | None,
    flock     : Flock,
    plan      : Plan
) -> dict[str, object]:
    """
    Moves `flock` through one step of `composite` in still air under
    Brownian increments drawn from the plan's seed, every drone having heard
    every other within the radio's range, each drone's zig reading its draw
    from `draws`, or from the key the step derives where `draws` is `None`.

    Returns:
        The radio's delay, the uniform draw each drone's zig reads, the
        flock the step reads, and its increments, beside the flock the step
        moves it to, rounded to four decimals.
    """
    air = Air(
        channels = zeros((4, 1, 1, 1, 1), dtype="float32"),
        names    = ("excess", "u", "v", "w"),
        origin   = (0.0, 0.0, 0.0, 0.0),
        spacing  = (1.0, 1.0, 1.0, 1.0)
    )
    increments = around(
        default_rng(plan.seed).normal(size=flock.forces.shape) * sqrt(plan.step_s),
        4
    )
    inbox = Inbox.from_broadcast(flock, composite.radio.range_m, flock)
    key   = seeded(plan.seed)
    still = jnp.zeros_like(flock.positions)
    t     = jnp.asarray(0.0)
    moved = stepped(
        air       = air,
        composite = composite,
        flock     = flock,
        inbox     = inbox,
        key       = key,
        path      = lambda *_: Flock(
            commands   = jnp.asarray(increments, dtype="float32"),
            forces     = still,
            positions  = still,
            spins      = still,
            velocities = still
        ),
        plan = plan,
        t    = t
    )
    if draws is None:
        draws = around(asarray(uniform(key, (len(increments),)), dtype=float), 4)
    else:
        moved = tree_at(
            lambda held: held.forces,
            moved,
            zigged(
                composite   = composite,
                draws       = draws,
                observation = Observation.sensed(
                    air    = air,
                    count  = plan.neighbors,
                    cruise = plan.cruise_m_s,
                    flock  = flock,
                    inbox  = inbox,
                    t      = t
                ),
                step_s = plan.step_s
            )
        )
    return {
        "delay_s"    : composite.radio.delay_s,
        "draws"      : draws.tolist(),
        "flock"      : rounded(flock),
        "increments" : increments.tolist(),
        "stepped"    : rounded(moved)
    }


def markdown_table(header: Collection[str], rows: Iterable[Iterable[str]]) -> str:
    """
    Lays out `rows` as a Markdown table beneath `header`, each heading in
    bold.
    """
    return "\n".join(
        [
            f"| {' | '.join(f'**{heading}**' for heading in header)} |",
            "|---" * len(header) + "|",
            *(f"| {' | '.join(row)} |" for row in rows)
        ]
    )


def paired(composite: Composite, plan: Plan) -> dict[str, object]:
    """
    Moves three flocks of twelve drones from `plan` through one step of
    `composite`, each with five zigs under way at its side, at its push
    back, and in its refractory time. The first sits astride the rim of the
    roost's disk on the side its course heads toward, its zigs read from
    the key the step derives, and the other two sit on the roost's center,
    half their drones drawing zero and the rest drawing across the chances a
    few zigging neighbors give, the last under a radio whose delay outlasts
    the zig's reaction and cue. Beside them, the radio carries a flock of
    eight more drones than a table holds on the roost's center, launched at
    half the spacing of the other three, so a drone hears more senders than
    its table keeps, its last drone flying back at the cruise speed from 5
    m past the radio's range ahead of the drone leading along the flock's
    mean velocity, so no drone holds it as the radio opens and it flies into
    the flock's range. A second run carries 120 drones scattered 25 m about
    the roost's center over a radio losing no packet, the even drones flying
    east at the cruise speed and the odd ones west, each velocity scattered
    1.5 m/s on every axis and its third drone flying on its second, and
    the curve orders 340 drones scattered the same way, its third drone on
    its second, each value drawn in double precision and rounded to four
    decimals.

    Returns:
        Each flock as `flown` lays it out, by its name, beside the run of
        the radio as `relayed` lays it out, the held senders at each packet
        of the second run, and the curve's drones beside the order `zorder`
        gives it.
    """
    zig            = composite.noise
    roost          = composite.roost
    rest           = zig.refractory_s
    forces         = zeros((12, 3))
    forces[:5, :2] = [
        [rest + zig.back_s + zig.side_s / 2, -1.0],
        [rest + zig.back_s + zig.side_s / 2, 1.0],
        [rest + zig.back_s / 2, -1.0],
        [rest + zig.back_s / 2, -1.0],
        [rest / 2, 1.0]
    ]
    draws       = zeros(len(forces))
    draws[1::2] = around(geomspace(1e-4, 4e-3, len(forces) // 2), 4)
    rim         = posed(
        center_m  = (roost.center_m[0] - roost.radius_m, roost.center_m[1]),
        composite = composite,
        forces    = forces,
        plan      = plan
    )
    middle = posed(
        center_m  = roost.center_m,
        composite = composite,
        forces    = forces,
        plan      = plan
    )
    radio = posed(
        center_m  = roost.center_m,
        composite = composite,
        forces    = zeros((Radio.slots + 8, 3)),
        plan      = plan,
        spacing_m = 4.0
    )
    heading = unit(radio.velocities.mean(0))
    leader  = radio.positions[(radio.positions @ heading).argmax()]
    late    = tree_at(
        lambda slots: slots.radio,
        composite,
        replace(composite.radio, delay_s=2 * (zig.reaction_s + zig.cue_s))
    )
    cloud             = default_rng(plan.seed)
    center            = (*roost.center_m, roost.height_m)
    points            = around(cloud.normal(center, 25.0, (340, 3)), 4)
    points[2]         = points[1]
    crossing          = around(cloud.normal(center, 25.0, (120, 3)), 4)
    crossing[2]       = crossing[1]
    velocities        = cloud.normal(0.0, 1.5, (120, 3))
    velocities[:, 0] += plan.cruise_m_s * (-1.0) ** arange(120)
    velocities        = around(velocities, 4)
    velocities[2]     = velocities[1]
    crossed           = relayed(
        flock = tree.map(
            lambda leaf: jnp.asarray(leaf, dtype="float32"),
            Flock(
                commands   = velocities,
                forces     = zeros((120, 3)),
                positions  = crossing,
                spins      = zeros((120, 3)),
                velocities = velocities
            )
        ),
        plan  = plan,
        radio = replace(composite.radio, loss=0.0)
    )
    return {
        "crossing" : {name: crossed[name] for name in ("flock", "held", "packets")},
        "curve"    : {
            "order"  : asarray(zorder(jnp.asarray(points))).tolist(),
            "points" : points.tolist()
        },
        "flocks"   : {
            "late"  : flown(
                composite = late,
                draws     = draws,
                flock     = middle,
                plan      = plan
            ),
            "rim"   : flown(
                composite = composite,
                draws     = None,
                flock     = rim,
                plan      = plan
            ),
            "roost" : flown(
                composite = composite,
                draws     = draws,
                flock     = middle,
                plan      = plan
            )
        },
        "radio"    : relayed(
            flock = tree_at(
                lambda flock: (flock.positions, flock.velocities),
                radio,
                (
                    radio.positions.at[-1].set(
                        jnp.round(leader + (composite.radio.range_m + 5) * heading, 4)
                    ),
                    radio.velocities.at[-1].set(
                        jnp.round(-plan.cruise_m_s * heading, 4)
                    )
                )
            ),
            plan  = plan,
            radio = composite.radio
        )
    }


def pieces(declared: Iterable[Piece]) -> str:
    """
    Lays out each of the kit's pieces in `declared` as a row naming the
    piece, the names a widget imports it by, the Zag.js machine it runs on,
    when a widget reads it, and what it replaces.
    """
    return markdown_table(
        header = [
            "Piece", "Imports", "Machine", "A Widget Reads It When", "It Replaces"
        ],
        rows = (
            [
                piece.name,
                ", ".join(f"`{name}`" for name in piece.exports),
                "None" if piece.machine is None else f"`{piece.machine}`",
                piece.when, piece.replaces
            ]
            for piece in declared
        )
    )


def posed(
    center_m  : Sequence[float],
    composite : Composite,
    forces    : ndarray,
    plan      : Plan,
    spacing_m : float = 8.0
) -> Flock:
    """
    Launches a flock of drones from `plan` at a launch spacing of
    `spacing_m` around `center_m`, one for each row of `forces`, which sets
    their zigs, each commanding a velocity apart from the one it flies and
    off the cruise speed and spinning toward the velocity it flies, its last
    drone moved half the cohesion's hard sphere ahead of the one before it
    along that drone's command. Every value is rounded to four decimals, so
    a copy of the step reads the same flock the package moves.
    """
    launch = plan.changed(
        center_m  = (*center_m, composite.roost.height_m),
        drones    = len(forces),
        spacing_m = spacing_m
    ).launched(seeded(plan.seed))
    commands      = asarray(launch.commands, dtype=float) * [0.8, 1.2, 0.5]
    positions     = asarray(launch.positions, dtype=float)
    positions[-1] = positions[-2] + (
        composite.cohesion.sphere_m / 2 * commands[-2] / linalg.norm(commands[-2])
    )
    return tree.map(
        lambda leaf: jnp.asarray(
            around(asarray(leaf, dtype=float), 4),
            dtype = "float32"
        ),
        Flock(
            commands  = commands,
            forces    = forces,
            positions = positions,
            spins     = cross(
                commands / linalg.norm(commands, axis=-1, keepdims=True),
                launch.velocities / linalg.norm(
                    launch.velocities,
                    axis     = -1,
                    keepdims = True
                )
            ),
            velocities = launch.velocities
        )
    )


def relayed(flock: Flock, plan: Plan, radio: Radio) -> dict[str, object]:
    """
    Carries the broadcasts of `flock` over `radio` from its opening for
    `radio.delay_s` and six packet periods past it, each drone flying its
    velocity, keeping at each step a packet falls on the uniform draw each
    pair's loss reads, by receiver and then by every other drone, unrounded
    so a copy keeps each packet the link keeps, and the senders each drone's
    table then holds.

    Returns:
        The forces, positions, and velocities of the flock the radio opens
        on, each step a packet falls at, and at each such step the draws,
        the senders each drone's table holds in the order of their indices,
        and the position it last heard each one at.
    """
    key    = seeded(plan.seed)
    drones = len(flock.positions)
    rows   = jnp.arange(drones)[:, None]
    others = jnp.broadcast_to(jnp.arange(drones), (drones, drones))[
        ~jnp.eye(drones, dtype=bool)
    ].reshape(drones, drones - 1)
    period      = radio.period_steps(plan.step_s)
    steps       = radio.delay_steps(plan.step_s) + 6 * period
    inbox, sent = radio.opened(flock, 0, plan.step_s)
    kept        = []
    for step in range(steps):
        inbox, sent = radio.heard(
            flock = tree_at(
                lambda flying: flying.positions,
                flock,
                flock.positions + step * plan.step_s * flock.velocities
            ),
            inbox  = inbox,
            key    = key,
            lag    = 0,
            sent   = sent,
            step   = jnp.asarray(step),
            step_s = plan.step_s
        )
        if step % period == 0:
            kept.append((radio.draws(key, rows, others, jnp.asarray(step)), inbox))

    def listed(held: Inbox) -> list[tuple[ndarray, ndarray]]:
        """
        Lists for each receiver the senders its table holds in `held`, in
        the order of their indices, beside the position it last heard each
        one at.
        """
        order     = jnp.where(held.heard, held.senders, drones).argsort(1)
        senders   = asarray(jnp.take_along_axis(held.senders, order, 1))
        positions = asarray(jnp.take_along_axis(held.positions, order[..., None], 1))
        return [
            (senders[receiver, :count], positions[receiver, :count])
            for receiver, count in enumerate(asarray(held.heard.sum(1)))
        ]

    tables = [listed(held) for _, held in kept]
    return {
        "draws"     : [asarray(draws, dtype=float).tolist() for draws, _ in kept],
        "flock"     : {
            name: leaf
            for name, leaf in rounded(flock).items()
            if name in ("forces", "positions", "velocities")
        },
        "held"      : [[row.tolist() for row, _ in table] for table in tables],
        "packets"   : list(range(0, steps, period)),
        "positions" : [
            [around(position.astype(float), 4).tolist() for _, position in table]
            for table in tables
        ]
    }


def rounded(state: Flock) -> dict[str, list]:
    """
    Lays out each array of `state` by its name, rounded to four decimals.
    """
    return {
        name: around(asarray(leaf, dtype=float), 4).tolist()
        for name, leaf in asdict(state).items()
    }


def specimens(identity: Identity) -> dict[str, str]:
    """
    Lays out each region of The Visual Identity, by the region's name:

    - `faces`, each face beside the text it sets, its weights, its styles,
      and the stack a reader falls back through
    - `heat`, each step of the heat ramp from light to dark
    - `marks`, each mark beside its file under each color scheme
    - `palette`, each role and each chart series beside the color it takes
      under each color scheme
    - `tagline`, the tagline as a quotation
    """
    palette = identity.palette
    roles   = [
        (name.title(), role, described(Palette, name))
        for name, role in palette
        if isinstance(role, Pair)
    ] + [
        (f"Chart {place}", role, f"Series {place} of a chart")
        for place, role in enumerate(palette.series, start=1)
    ]
    return {
        "faces"   : markdown_table(
            header = ["Face", "Sets", "Weights", "Styles", "Stack"],
            rows   = (
                [
                    face.family,
                    f"*{described(type(identity.faces), kind)}*",
                    ", ".join(map(str, face.weights)),
                    "Roman and italic" if face.italic else "Roman",
                    f"`{', '.join([f"'{face.family}'", *face.fallbacks])}`"
                ]
                for kind, face in identity.faces
            )
        ),
        "heat"    : markdown_table(
            header = ["Step", "Color"],
            rows   = (
                [str(step), colored(name, palette)]
                for step, name in enumerate(palette.heat, start=1)
            )
        ),
        "marks"   : markdown_table(
            header = ["Mark", "Light", "Dark", "Holds"],
            rows   = (
                [
                    kind.title(),
                    f"`{pair.light}`",
                    f"`{pair.dark}`",
                    f"*{described(type(identity.marks), kind)}*"
                ]
                for kind, pair in identity.marks
            )
        ),
        "palette" : markdown_table(
            header = ["Role", "Light", "Dark", "Holds"],
            rows   = (
                [
                    label,
                    colored(role.light, palette),
                    colored(role.dark, palette),
                    f"*{holds}*"
                ]
                for label, role, holds in roles
            )
        ),
        "tagline" : f'> *"{identity.tagline.text}"*'
    }


def tabled(kind: type) -> str:
    """
    Lays out each field `kind` declares as a row of a Markdown table, in the
    order the class declares them, the name and the default in code beside
    the description, with a column of bounds only where a field sets one.
    """
    schema = TypeAdapter(kind).json_schema()
    rows   = list(fielded(schema.get("$defs", {}), schema["properties"]))
    bounds = any(bound for _, _, bound, _ in rows)
    return markdown_table(
        header = ["Field", "Default", *["Bound"] * bounds, "Holds"],
        rows   = (
            [f"`{name}`", f"`{default}`", *[bound] * bounds, holds]
            for name, default, bound, holds in rows
        )
    )


def zigged(
    composite   : Composite,
    draws       : ndarray,
    observation : Observation,
    step_s      : float
) -> Array:
    """
    Advances each drone's zig by one step of `step_s` as `Composite.excited`
    does, from what each drone sensed in `observation` and the uniform draw
    `draws` holds for it in place of one from a key.
    """
    return vmap(
        partial(composite.noise.excited, delay_s=composite.radio.delay_s, step_s=step_s)
    )(
        draw        = jnp.asarray(draws, dtype="float32"),
        observation = observation,
        turn        = vmap(composite.roost)(observation)
    )
