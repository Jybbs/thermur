"""
Pins what the render task writes and how it reads the copies the
repository holds:

- The identity as JSON, the crop with the shape of the sample as JSON, the
  threat with the constants of its air as JSON, the conditions of each trial
  as JSON, and each page carrying the definition of every study it cites and
  each region written from its record
- A drifted or missing copy reported, and only a drifted copy written again
- Each marker naming no entry listed on its page, and each region missing
  its pair of markers refused
- The table a region lays a record's fields out as, and the specimens The
  Visual Identity's regions lay the identity out as
"""

from common.edits import append, edit
from json         import loads
from pathlib      import Path
from pydantic     import BaseModel, Field
from typing       import Annotated

from thermur.controller.components import SoftCore
from thermur.docs.outputs          import Outputs, markdown_table, specimens, tabled
from thermur.environment.schemas import (
    BASE_K,
    Crop,
    GRAVITY_M_S2,
    HEAT_CAPACITY_J_KG_K,
    ParametricThreat,
    Ranges,
    Sample
)
from thermur.identity.schemas import Identity
from thermur.metrics.schemas  import Plan
from thermur.metrics.trials   import Trial
from thermur.repo.checkout    import Checkout
from thermur.repo.schemas     import Finding

# `DATA`, `IDENTITY`, and `RULES` hold the pages whose regions the render
# writes, from the fire data's records, the identity's, and the flight's.
DATA     = Path("docs/pages/data/index.md")
IDENTITY = Path("docs/pages/reference/identity.md")
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
    readme  = (checkout.root / "README.md").stat().st_mtime_ns
    (checkout.root / outputs.identity).unlink()
    append(checkout, str(page), "\nTurns travel[^attanasi2014].\n")

    assert outputs.drifted == [outputs.identity, page]
    assert outputs.write() == [outputs.identity, page]
    assert outputs.write() == []
    assert (checkout.root / "README.md").stat().st_mtime_ns == readme
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


def test_an_edited_region_drifts_and_is_written_again(checkout: Checkout):
    """
    Asserts that a region edited by hand drifts and is written back from its
    record, after which the page no longer drifts.
    """
    edit(
        checkout,
        str(DATA),
        new = "| `archive` | `edited.tar.gz` |",
        old = "| `archive` | `samples.tar.gz` |"
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
            "`docs/references.bib`"
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
    Asserts that The Interaction Rules carries the fields of the plan and
    the contact, and for each open slot those of its default and then each
    candidate, every field named after its component.
    """
    regions = Outputs(checkout=checkout).regions[RULES]
    rows    = regions["cohesion"].splitlines()[2:]

    assert regions["contact"] == tabled(SoftCore)
    assert regions["plan"] == tabled(Plan)
    assert set(regions) == {
        "contact",
        "plan",
        *(trial.slot for trial in Trial.listed())
    }
    assert [row.split("`")[1].partition(".")[0] for row in rows] == [
        *["morse"] * 4,
        "centroid",
        *["elastic"] * 2,
        *["fitted"] * 4
    ]
