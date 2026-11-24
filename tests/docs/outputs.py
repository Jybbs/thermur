"""
Pins what the render task writes and how it reads the copies the
repository holds:

- The identity as JSON, the crop with the shape of the sample as JSON, the
  threat with the constants of its air as JSON, the conditions of each trial
  as JSON, the default slots and the plan the site's flocks fly as JSON
  beside one step of three small flocks under them and a run of the radio,
  the zig drawn in place of the key moving as the step's own, and each page
  carrying the definition of every study it cites and each region written
  from its record
- A drifted or missing copy reported, and only a drifted copy written again
- Each marker naming no entry listed on its page, and each region missing
  its pair of markers refused
- The table a region lays a record's fields out as, and the specimens The
  Visual Identity's regions lay the identity out as
"""

from common.edits import append, edit
from dataclasses  import asdict, fields
from jax          import numpy as jnp
from jax.random   import key as seeded, uniform
from json         import loads
from numpy        import asarray, linalg, minimum
from pathlib      import Path
from pydantic     import BaseModel, Field
from typing       import Annotated

from thermur.controller.components import HardSphere, Home
from thermur.controller.schemas    import Composite, Flock, Radio
from thermur.docs.kit              import Piece
from thermur.docs.outputs          import Outputs, flown, markdown_table, normed, pieces, specimens, tabled
from thermur.environment.plant     import Plant
from thermur.environment.schemas import (
    BASE_K,
    Crop,
    GRAVITY_M_S2,
    HEAT_CAPACITY_J_KG_K,
    ParametricThreat,
    Ranges,
    Sample
)
from thermur.identity.schemas  import Identity
from thermur.metrics.norms     import Norm, Norms, Spread
from thermur.metrics.schemas   import Plan
from thermur.metrics.trials    import Trial
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout
from thermur.repo.schemas      import Finding

# `DATA`, `IDENTITY`, `KIT`, and `RULES` hold the pages whose regions the
# render writes, from the fire data's records, the identity's, the kit's,
# and the flight's.
DATA     = Path("docs/pages/data/index.md")
IDENTITY = Path("docs/pages/reference/identity.md")
KIT      = Path("docs/pages/reference/kit.md")
RULES    = Path("docs/pages/methods/rules.md")


class Inner(BaseModel, use_attribute_docstrings=True):
    """
    A record another record nests.
    """

    depth_m: Annotated[float, Field(gt=0, lt=10)] = 1.0
    """
    The depth, in meters.
    """


class Unbounded(BaseModel, use_attribute_docstrings=True):
    """
    A record whose one field sets no bound.
    """

    label: str | None = None
    """
    The label.
    """


class Outer(BaseModel, use_attribute_docstrings=True):
    """
    A record nesting another beside a bounded field and a plain one.
    """

    count: Annotated[int, Field(ge=1, le=5)] = 2
    """
    The count, on
    two lines.
    """

    inner: Inner = Inner()
    """
    The nested record.
    """

    name: str = "plain"
    """
    The name.
    """


def test_a_missing_or_edited_copy_drifts_and_is_written_again(checkout: Checkout):
    """
    Asserts that a missing identity and a page citing a study it does not
    define both count as drifted, that writing rewrites each while leaving a
    current page untouched, and that a second write finds nothing to write.
    """
    outputs = Outputs(checkout=checkout)
    page    = Path("docs/pages/index.md")
    readme  = (checkout.root / ".github/README.md").stat().st_mtime_ns
    (checkout.root / outputs.identity).unlink()
    append(checkout, str(page), "\nTurns travel[^attanasi2014].\n")

    assert outputs.drifted == [outputs.identity, page]
    assert outputs.write() == [outputs.identity, page]
    assert outputs.write() == []
    assert (checkout.root / ".github/README.md").stat().st_mtime_ns == readme
    assert (
        checkout.read(page)
                .endswith(f"\n\n{checkout.bibliography.cite('attanasi2014')}\n")
    )


def test_a_missing_page_refuses_each_of_its_regions(checkout: Checkout):
    """
    Asserts that every region of a page the checkout no longer holds is
    refused, in the order the page's regions are declared.
    """
    (checkout.root / DATA).unlink()

    assert [finding.message for finding in Outputs(checkout=checkout).refused] == [
        f"`{DATA}` does not carry the opening and then the closing marker of its "
        f"`{name}` region once each, so `mise render` leaves the region as it stands"
        for name in ("crop", "ranges", "sample", "threat")
    ]


def test_a_norm_row_names_each_part_and_words_an_unstated_one():
    """
    Asserts that each norm lays out as a row naming its statistic, its
    figure with its unit or its lower bound, its spread with the spread's
    kind, its flocks, its system, and a link to its DOI, a spread or a count
    of flocks the source leaves out written as words so no cell is empty.
    """
    table = normed(
        Norms(
            {
                "border_density" : Norm(
                    above  = True,
                    doi    = "10.1/a",
                    figure = 1.0,
                    system = "starlings",
                    unit   = ""
                ),
                "wave_speed_m_s" : Norm(
                    doi    = "10.1/b",
                    figure = 30.0,
                    flocks = 12,
                    kind   = Spread.RANGE,
                    spread = 10.0,
                    system = "starlings",
                    unit   = "m/s"
                )
            }
        )
    )

    assert [row.strip("| ").split(" | ") for row in table.splitlines()[2:]] == [
        [
            "`border_density`", "above 1", "none stated",
            "not stated", "starlings", "[10.1/a](https://doi.org/10.1/a)"
        ],
        [
            "`wave_speed_m_s`", "30 m/s", "±10, half range",
            "12", "starlings", "[10.1/b](https://doi.org/10.1/b)"
        ]
    ]


def test_a_page_git_ignores_is_left_alone(checkout: Checkout):
    """
    Asserts that a Markdown file under a folder `.gitignore` names, such
    as a package's readme under `docs/node_modules/`, is neither read nor
    written, however it cites.
    """
    ignored = checkout.root / "docs/node_modules/package/README.md"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("A study[^smith2020] and another[^cavagna2010].\n")

    outputs = Outputs(checkout=checkout)

    assert (outputs.drifted, outputs.unknown) == ([], [])


def test_a_region_missing_its_pair_of_markers_is_refused_and_left(checkout: Checkout):
    """
    Asserts that a region whose page carries no closing marker is refused,
    naming the page and the region, while the render leaves it as it stands
    and writes the page's other regions.
    """
    edit(checkout, str(DATA), new="", old="<!-- /sample -->")
    edit(checkout, str(DATA), new="| `floor_m` | `0.0` |", old="| `floor_m` | `20.0` |")
    outputs = Outputs(checkout=checkout)

    assert outputs.refused == [
        Finding(
            file    = DATA,
            message = f"`{DATA}` does not carry the opening and then the closing "
            "marker of its `sample` region once each, so `mise render` leaves the "
            "region as it stands"
        )
    ]
    assert outputs.write() == [DATA]
    assert f"\n\n{tabled(Crop)}\n\n" in checkout.read(DATA)
    assert "<!-- /sample -->" not in checkout.read(DATA)


def test_a_table_bolds_each_heading_above_its_rows():
    """
    Asserts that a table sets each heading in bold above a divider of one
    cell per heading and each row in the order given.
    """
    assert markdown_table(
        ["Step", "Color"],
        [["1", "red"], ["2", "blue"]]
    ) == "\n".join(
        [
            "| **Step** | **Color** |", "|---|---|", "| 1 | red |", "| 2 | blue |"
        ]
    )


def test_a_table_names_each_field_beside_its_default_bound_and_description():
    """
    Asserts that a table lays out each field in the order its class declares
    them, a nested record's fields under its name and a dot, each beside its
    default and its bounds in words, lower before upper, and its description
    on one line without its closing period.
    """
    assert tabled(Outer) == "\n".join(
        [
            "| **Field** | **Default** | **Bound** | **Holds** |", "|---|---|---|---|",
            "| `count` | `2` | at least 1 and at most 5 | The count, on two lines |",
            "| `inner.depth_m` | `1.0` | above 0 and below 10 | The depth, in meters |",
            "| `name` | `plain` |  | The name |"
        ]
    )


def test_a_table_of_fields_setting_no_bound_carries_no_bound_column():
    """
    Asserts that a table of fields none of which sets a bound leaves the
    bound column out.
    """
    assert tabled(Unbounded) == "\n".join(
        [
            "| **Field** | **Default** | **Holds** |", "|---|---|---|",
            "| `label` | `None` | The label |"
        ]
    )


def test_a_zig_drawn_in_place_of_the_key_steps_as_the_composite_does():
    """
    Asserts that one step of the rim's flock under the draws the step's key
    gives, read in place of the key, moves each zig as the composite's own
    step does.
    """
    settings  = Settings()
    composite = Composite.from_settings(settings)
    plan      = settings.plan
    rim       = loads(Outputs().written[Outputs().parity])["flocks"]["rim"]["flock"]
    flock     = Flock(
        **{name: jnp.asarray(value, dtype="float32") for name, value in rim.items()}
    )
    drawn = asarray(uniform(seeded(plan.seed), (len(rim["forces"]),)), dtype=float)

    assert flown(composite, drawn, flock, plan)["stepped"] == flown(
        composite,
        None,
        flock,
        plan
    )["stepped"]


def test_an_edited_region_drifts_and_is_written_again(checkout: Checkout):
    """
    Asserts that a region edited by hand drifts and is written back from its
    record, after which the page no longer drifts.
    """
    edit(
        checkout,
        str(DATA),
        new = "| `file` | `edited.nc` |",
        old = "| `file` | `wrf_sample.nc` |"
    )

    assert Outputs(checkout=checkout).write() == [DATA]
    assert f"\n\n{tabled(Sample)}\n\n" in checkout.read(DATA)
    assert Outputs(checkout=checkout).drifted == []


def test_each_marker_naming_no_entry_is_listed_on_its_page(checkout: Checkout):
    """
    Asserts that a marker under a key the bibliography lacks is listed with
    its page and its line, on a page git would track though nothing has
    added it yet, and that the page's other citations still drift until
    written.
    """
    page = Path("docs/pages/notes.md")
    (
        checkout.root / page
    ).write_text("# Notes\n\nA study[^smith2020] beside[^cavagna2010].\n")
    outputs = Outputs(checkout=checkout)

    assert outputs.unknown == [
        Finding(
            file    = page,
            message = f"Line 3 of `{page}` cites `smith2020`, which names no entry in "
            "`docs/pages/references.bib`"
        )
    ]
    assert outputs.drifted == [page]


def test_each_region_is_written_from_its_record(checkout: Checkout):
    """
    Asserts that the Data page carries the fields `Sample`, `Crop`,
    `Ranges`, and `ParametricThreat` declare, and that every region of every
    page sits between its pair of markers as the render writes it, so the
    copies the repository holds have not drifted.
    """
    outputs = Outputs(checkout=checkout)

    assert outputs.regions[DATA] == {
        "crop"   : tabled(Crop),
        "ranges" : tabled(Ranges),
        "sample" : tabled(Sample),
        "threat" : tabled(ParametricThreat)
    }
    assert all(
        f"<!-- {name} -->\n\n{body}\n\n<!-- /{name} -->" in checkout.read(page)
        for page, regions in outputs.regions.items()
        for name, body in regions.items()
    )
    assert not {DATA, IDENTITY, RULES} & set(outputs.drifted)


def test_the_identity_page_lays_out_each_specimen_from_its_record(checkout: Checkout):
    """
    Asserts that The Visual Identity carries the specimens of the shipped
    identity, each role of the palette before each chart series, each heat
    step from the lightest, each face beside its styles, each mark beside
    its files, and the tagline as a quotation.
    """
    identity = Identity.shipped()
    regions  = Outputs(checkout=checkout).regions[IDENTITY]
    palette  = regions["palette"].splitlines()[2:]

    assert regions == specimens(identity)
    assert [row.split(" | ")[0].removeprefix("| ") for row in palette] == [
        "Critical", "Ground", "Ink", "Lead", "Warning",
        *(f"Chart {place}" for place in range(1, len(identity.palette.series) + 1))
    ]
    assert regions["heat"].splitlines()[2] == (
        f"| 1 | Kindleflame `{identity.palette.colors['kindleflame']}` |"
    )
    assert "| Bellefair | " in regions["faces"]
    assert "| Roman |" in regions["faces"]
    assert "| Roman and italic |" in regions["faces"]
    assert (
        f"| Small | `{identity.marks.small.light}` | `{identity.marks.small.dark}` |"
        in regions["marks"]
    )
    assert regions["tagline"] == f'> *"{identity.tagline.text}"*'


def test_the_kit_page_lays_out_each_piece_from_its_declaration(checkout: Checkout):
    """
    Asserts that The Kit carries a row per piece declared in the clone, each
    import and machine backticked and a piece no machine covers marked as
    running on none.
    """
    declared = Piece.declared(checkout.root)
    regions  = Outputs(checkout=checkout).regions[KIT]
    table    = regions["pieces"]
    rows     = table.splitlines()[2:]

    assert regions == {"pieces": pieces(declared)}
    assert len(rows) == len(declared)
    assert "| Slider | `Slider` | `slider` | " in table
    assert "| The Gate | `onMotion`, `pause`, `paused`, `still` | None | " in table
    assert any(" | None | " in row for row in rows)


def test_the_render_writes_each_default_slot_and_the_plan_the_site_flies(
    checkout: Checkout
):
    """
    Asserts that the composite's file holds the kind and the fields of each
    slot's default scaled to the plan, and the plan's fields a site flock
    reads, and that the copy the repository holds has not drifted.
    """
    outputs  = Outputs(checkout=checkout)
    settings = Settings()
    written  = loads(outputs.written[outputs.composite])

    assert {slot: entry.pop(
        "kind"
    ) for slot, entry in written.items() if slot != "plan"} == {
        slot.name: type(getattr(settings, slot.name)).__name__ for slot in fields(
            Composite
        )
    }
    assert written["contact"] == {
        name: float(value)
        for name, value in asdict(Composite.from_settings(settings).contact).items()
    }
    assert written["plan"] == settings.plan.model_dump(
        include = set(written["plan"]),
        mode    = "json"
    )
    assert outputs.composite not in outputs.drifted


def test_the_render_writes_each_trial_s_conditions_by_slot(checkout: Checkout):
    """
    Asserts that the trials' file holds what each condition of each trial
    sets, by slot, so the charts place each mean where the trial flew it,
    and that the copy the repository holds has not drifted.
    """
    outputs = Outputs(checkout=checkout)

    assert loads(outputs.written[outputs.trials]) == {
        trial.slot: trial.conditions for trial in Trial.listed()
    }
    assert loads(outputs.written[outputs.trials])["coupling"] == {
        "dense"  : {"spacing_m": 10.0},
        "middle" : {"spacing_m": 15.0},
        "sparse" : {"spacing_m": 25.0}
    }
    assert outputs.trials not in outputs.drifted


def test_the_render_writes_one_step_of_three_small_flocks_and_a_run_of_the_radio(
    checkout: Checkout
):
    """
    Asserts that the parity file holds a flock of twelve astride the rim
    of the roost's disk and two on its center, each commanding off its
    velocity and off the cruise speed with one pair inside the cohesion's
    hard sphere, the center's resting drones starting a zig toward their
    zigging neighbors' side where their draw is zero and the last flock's
    radio delay outlasting the zig's reaction and cue, beside a run of the
    radio in which each drone's table holds `Radio.slots` of the senders in
    range as it opens, or every one where fewer are, some drone hearing more
    senders than its table holds, and the last drone, which no drone holds
    as it opens, held by a drone by the last packet, a run over a crossing
    flock whose tables turn over and whose third drone sits on its second,
    a curve over drones with one coincident pair, and that the copy the
    repository holds has not drifted.
    """
    outputs   = Outputs(checkout=checkout)
    settings  = Settings()
    composite = Composite.from_settings(settings)
    zig       = composite.noise
    paired    = loads(outputs.written[outputs.parity])
    flocks    = {
        name: {part: asarray(value) for part, value in flock["flock"].items()}
        | {
            "draws"   : asarray(flock["draws"]),
            "stepped" : asarray(flock["stepped"]["forces"])
        }
        for name, flock in paired["flocks"].items()
    }
    rim      = flocks["rim"]
    roost    = flocks["roost"]
    outward  = linalg.norm(rim["positions"][:, :2] - composite.roost.center_m, axis=-1)
    apart    = linalg.norm(rim["positions"][:, None] - rim["positions"], axis=-1)
    resting  = (roost["forces"][:, 0] == 0) & (roost["draws"] == 0)
    radio    = paired["radio"]
    crossing = paired["crossing"]
    curve    = paired["curve"]
    sent     = asarray(radio["flock"]["positions"])
    within   = (
        linalg.norm(sent[:, None] - sent, axis=-1) <= composite.radio.range_m
    ).sum(1) - 1

    assert outward.min() < composite.roost.radius_m < outward.max()
    assert (rim["commands"] != rim["velocities"]).all()
    assert (
        abs(linalg.norm(rim["commands"], axis=-1) - settings.plan.cruise_m_s) > 1
    ).all()
    assert apart[apart > 0].min() < composite.cohesion.sphere_m
    assert (roost["stepped"][resting, 0] > zig.refractory_s).all()
    assert (roost["stepped"][resting, 1] == -1).any()
    assert paired["flocks"]["late"]["delay_s"] > zig.reaction_s + zig.cue_s
    assert (within > Radio.slots).any()
    assert list(map(len, radio["held"][0])) == minimum(within, Radio.slots).tolist()
    assert not any(len(sent) - 1 in row for row in radio["held"][0])
    assert any(len(sent) - 1 in row for row in radio["held"][-1])
    assert (
        len(radio["packets"])
        == len(radio["draws"])
        == len(radio["held"])
        == len(radio["positions"])
    )
    assert crossing["held"][0] != crossing["held"][-1]
    assert crossing["flock"]["positions"][1] == crossing["flock"]["positions"][2]
    assert curve["points"][1] == curve["points"][2]
    assert sorted(curve["order"]) == list(range(len(curve["points"])))
    assert outputs.parity not in outputs.drifted


def test_the_render_writes_the_crop_beside_the_shape_of_the_sample():
    """
    Asserts that the crop's file holds the record's heights and the sample's
    length and snapshot times, so the sketch and the record read one crop.
    """
    outputs = Outputs()

    assert loads(outputs.written[outputs.crop]) == {
        "floor_m"   : Crop().floor_m,
        "length_m"  : Sample().length_m,
        "levels"    : Crop().levels,
        "spacing_m" : Crop().spacing_m,
        "times_s"   : list(Sample().times_s)
    }


def test_the_render_writes_the_identity_as_json():
    """
    Asserts that the identity file holds the identity as `Identity.shipped`
    reads it, so the site reads the very records the chart theme does.
    """
    outputs = Outputs()

    assert loads(outputs.written[outputs.identity]) == Identity.shipped().model_dump(
        mode = "json"
    )


def test_the_render_writes_the_threat_beside_the_constants_of_its_air():
    """
    Asserts that the threat's file holds the default record and the
    constants the dataset's air is worked out from, so the drawing and the
    dataset read one front.
    """
    outputs = Outputs()

    assert loads(outputs.written[outputs.threat]) == {
        "base_k"               : BASE_K,
        "gravity_m_s2"         : GRAVITY_M_S2,
        "heat_capacity_j_kg_k" : HEAT_CAPACITY_J_KG_K
    } | ParametricThreat().model_dump()


def test_the_rules_page_tables_each_slot_s_default_beside_its_candidates(
    checkout: Checkout
):
    """
    Asserts that The Interaction Rules carries the fields of the plan, the
    contact, the plant, the radio, and the roost, and for each open slot
    those of its default and then each candidate, every field named after
    its component.
    """
    regions = Outputs(checkout=checkout).regions[RULES]
    rows    = regions["cohesion"].splitlines()[2:]

    assert regions["contact"] == tabled(HardSphere)
    assert regions["plan"] == tabled(Plan)
    assert regions["plant"] == tabled(Plant)
    assert regions["radio"] == tabled(Radio)
    assert regions["roost"] == tabled(Home)
    assert set(regions) == {
        "contact",
        "plan",
        "plant",
        "radio",
        "roost",
        *(trial.slot for trial in Trial.listed())
    }
    assert [row.split("`")[1].partition(".")[0] for row in rows] == [
        *["centered"] * 3,
        "centroid",
        *["elastic"] * 2,
        *["fitted"] * 4
    ]
    assert [
        row.split("`")[1].partition(".")[0]
        for row in regions["speed"].splitlines()[2:]
    ] == [*["cruise"] * 3, *["marginal"] * 3]
