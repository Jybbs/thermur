"""
Holds `Outputs`, the files the `docs:render` task writes into the site
and every Markdown page git tracks, which `repo:audit` holds the committed
copies to, beside `tabled`, which lays a record's fields out as the table a
page's region holds.
"""

from collections.abc import Iterator, Mapping
from functools       import cached_property
from json            import dumps
from pathlib         import Path
from pydantic        import BaseModel, Field, TypeAdapter, create_model

from thermur.docs.pages          import Page
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

    crop: Path = Path("docs/.vitepress/lib/rendered/crop.json")
    """
    The file the Data page's sketch of the part kept reads the crop's defaults
    and the sample's length and snapshot times from.
    """

    identity: Path = Path("docs/.vitepress/lib/rendered/identity.json")
    """
    The file the site's theme and its preview cards read the identity from.
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
        Data page, and on The Interaction Rules the fields `Plan` and the
        contact declare beside those of each open slot's default and each of
        its candidates, every one named for its component.
        """
        defaults = Settings()
        return {
            Path("docs/pages/data/index.md")    : {
                "crop"   : tabled(Crop),
                "ranges" : tabled(Ranges),
                "sample" : tabled(Sample),
                "threat" : tabled(ParametricThreat)
            },
            Path("docs/pages/methods/rules.md") : {
                "contact" : tabled(type(defaults.contact)),
                "plan"    : tabled(Plan)
            } | {
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
        sample = Sample().model_dump(include={"length_m", "times_s"})
        threat = {
            "base_k"               : BASE_K,
            "gravity_m_s2"         : GRAVITY_M_S2,
            "heat_capacity_j_kg_k" : HEAT_CAPACITY_J_KG_K
        } | ParametricThreat().model_dump()
        return {
            self.crop     : dumps(crop | sample, indent=2) + "\n",
            self.identity : Identity.shipped().model_dump_json(indent=2) + "\n",
            self.threat   : dumps(threat, indent=2) + "\n",
            self.trials   : dumps(
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


def tabled(kind: type) -> str:
    """
    Lays out each field `kind` declares as a row of a Markdown table, in the
    order the class declares them, the name and the default in code beside
    the description, with a column of bounds only where a field sets one.
    """
    schema = TypeAdapter(kind).json_schema()
    rows   = list(fielded(schema.get("$defs", {}), schema["properties"]))
    bounds = any(bound for _, _, bound, _ in rows)
    header = ["**Field**", "**Default**", *["**Bound**"] * bounds, "**Holds**"]
    return "\n".join(
        [
            f"| {' | '.join(header)} |",
            "|---" * len(header) + "|",
            *(
                f"| `{name}` | `{default}` | {f'{bound} | ' * bounds}{holds} |"
                for name, default, bound, holds in rows
            )
        ]
    )
