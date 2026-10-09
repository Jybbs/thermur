"""
Holds the records the repository audit reads and reports, meaning the
`Manifest` with its `Author` rows, the `Config`, and the `Labels` with their
`Label` rows validated out of the checkout's TOML, and the `Release` with
its `Category` rows, each `Template`, and the `Citation` with its `Person`
rows validated out of its YAML, all through their `Document` base. Beside
them sit each `Task` mise lists and each `Finding` a check makes beside each
`Parity` a check compares.
"""

from collections            import Counter
from collections.abc        import Iterable, Sequence
from operator               import attrgetter
from packaging.requirements import Requirement
from pathlib                import Path
from pydantic               import (
    AfterValidator,
    AliasPath,
    BaseModel,
    BeforeValidator,
    Field,
    PlainValidator,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator
)
from re      import search
from tomllib import loads
from typing  import Annotated, Self
from yaml_rs import loads as from_yaml

type Dependency = Annotated[Requirement, PlainValidator(Requirement)]
type Minor      = Annotated[str, AfterValidator(read_minor)]


class Author(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One author `[project].authors` names.
    """

    name: str
    """
    The author's name as a copyright line gives it.
    """


def read_commands(run: Iterable[str | dict]) -> list[str]:
    """
    Keeps the lines a shell runs out of the `run` mise lists for a task,
    leaving out each `task` or `tasks` table, which names another task mise
    runs at that point.
    """
    return [line for line in run if isinstance(line, str)]


def read_labels(labels: str | list[str]) -> list[str]:
    """
    Reads the labels an issue template lists, which GitHub takes as a list
    or as one string separating them by commas.
    """
    if isinstance(labels, str):
        return list(filter(None, map(str.strip, labels.split(","))))

    return labels


def read_minor(version: str) -> str:
    """
    Reads the first `<major>.<minor>` in `version`, so `>=3.14` and `3.14.8`
    both read as `3.14`.

    Raises:
        ValueError: Where `version` names no major and minor version, which
                    validation reports against the field.
    """
    if match := search(r"\d+\.\d+", version):
        return match[0]

    raise ValueError("names no major and minor version")


class Category(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One release-notes category `.github/release.yml` declares.
    """

    labels: list[str]
    """
    The labels any one of which files a pull request under the category,
    where `*` files every pull request no earlier category takes.
    """


class Document(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    A TOML or YAML file of the checkout, declaring the fields the checks
    read and validated from the file's text, whose validation error carries
    the file's path from the root as its title.
    """

    @property
    def file(self) -> Path:
        """
        Reads the path of the file a titled model validates, relative to the
        root, out of its title.
        """
        return Path(self.model_config["title"])

    @staticmethod
    def load(suffix: str, text: str) -> object:
        """
        Reads `text`, the text of a file whose name ends on `suffix`, as
        TOML where the suffix is `.toml`, as the YAML front matter opening
        a Markdown file where it is `.md`, and as YAML otherwise, keeping a
        YAML timestamp as its text.

        Raises:
            ValueError: Where the TOML or the YAML fails to parse, or where
                        a Markdown file opens on no front matter, which
                        validation reports against the whole document.
        """
        if suffix == ".toml":
            return loads(text)

        if suffix == ".md":
            if front := search(r"(?ms)\A---\n(.*?)^---$", text):
                return from_yaml(front[1], parse_datetime=False)

            raise ValueError("opens on no front matter")

        return from_yaml(text, parse_datetime=False)

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: object, info: ValidationInfo) -> object:
        """
        Reads text through `load` under the suffix the context names,
        passing a mapping already read through unchanged.
        """
        if not isinstance(data, str):
            return data

        return cls.load(info.context["suffix"], data)

    @classmethod
    def read(cls, root: Path, path: Path | None = None) -> Self:
        """
        Validates the file at `path` under `root`, or at the path the title
        names where `path` is `None`.

        Raises:
            ValidationError: Titled with the file's path, where the file
                             fails to parse or to validate.
        """
        path = Path(path or cls.model_config["title"])
        try:
            return cls.model_validate(
                (root / path).read_text(encoding="utf-8"),
                context = {"suffix": path.suffix}
            )
        except ValidationError as error:
            raise ValidationError.from_exception_data(
                str(path),
                error.errors()
            ) from None


class Config(Document, title=".mise/config.toml"):
    """
    The tools `.mise/config.toml` pins and the folder its `[env]` puts on
    the path.
    """

    bin: Path = Field(validation_alias=AliasPath("env", "_", "path"))
    """
    The folder `_.path` puts on the path, relative to the root.
    """

    python: Minor = Field(validation_alias=AliasPath("tools", "python"))
    """
    The major and minor of the `python` mise pins.
    """

    uv: str = Field(validation_alias=AliasPath("tools", "uv"))
    """
    The `uv` release mise pins.
    """


class Finding(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One place the checkout breaks an invariant the audit holds it to, at the
    file a reader opens to put it right.
    """

    file: Path
    """
    The file the finding sits in, relative to the root of the checkout.
    """

    message: str
    """
    What is wrong, in a sentence a reader acts on.
    """

    @property
    def annotation(self) -> str:
        """
        Writes the finding as the GitHub Actions error annotation the runner
        shows on `file`, percent-encoding each line break and `%`, and each
        `,` and `:` in the file name, which a workflow command reads as its
        own syntax.
        """
        escapes = {"%": "%25", "\n": "%0A", "\r": "%0D"}
        place   = str(self.file).translate(
            str.maketrans(escapes | {",": "%2C", ":": "%3A"})
        )
        return f"::error file={place}::{self.message.translate(str.maketrans(escapes))}"


class Label(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One label `.github/labels.toml` declares, as GitHub shows it on an issue
    or a pull request.
    """

    color: Annotated[str, Field(pattern=r"^[0-9a-f]{6}$")]
    """
    The label's color as six lowercase hex digits.
    """

    description: str
    """
    What the label covers, in one line.
    """

    name: str
    """
    The label's glyph and word, as an issue or a pull request carries it.
    """

    @property
    def command(self) -> list[str]:
        """
        Writes the `gh label create` command that creates the label on the
        repository the checkout's remote names, or updates the color and the
        description of the label already carrying its name.
        """
        return [
            "gh", "label", "create", "--color", self.color,
            "--description", self.description, "--force", self.name
        ]


class Labels(Document, extra="forbid", title=".github/labels.toml"):
    """
    The labels `.github/labels.toml` declares, the one place a label is
    written down, holding nothing beside its rows.
    """

    rows: list[Label] = Field(validation_alias=AliasPath("labels"))
    """
    The labels, in the order the registry lists them.
    """

    @property
    def names(self) -> list[str]:
        """
        Lists the name of each label, in the order the registry lists them.
        """
        return [label.name for label in self.rows]

    def omits(self, live: Iterable[str]) -> list[str]:
        """
        Keeps each of the `live` label names the registry declares no label
        under, in the order `live` gives them.
        """
        return [name for name in live if name not in self.names]

    @field_validator("rows")
    @classmethod
    def reject_repeats(cls, rows: list[Label]) -> list[Label]:
        """
        Passes `rows` through where no two labels share a name or a color.

        Raises:
            ValueError: Naming each name and each color two or more labels
                        share, which validation reports against the field.
        """
        if repeats := [
            f"the {field} `{value}`"
            for field in ("name", "color")
            for value, count in Counter(map(attrgetter(field), rows)).items()
            if count > 1
        ]:
            raise ValueError(f"declares {join_names(repeats)} more than once")

        return rows


class Citation(Document, title="CITATION.cff"):
    """
    The declarations `CITATION.cff` restates from the manifest, each `None`
    where the file leaves it out.
    """

    authors: list[Person] = []
    """
    The people the citation credits, in the order it names them.
    """

    license: str | None = None
    """
    The SPDX expression the citation declares.
    """

    repository: str | None = Field(None, validation_alias="repository-code")
    """
    The address of the repository holding the code.
    """

    version: str | None = None
    """
    The release the citation cites.
    """

    @property
    def holders(self) -> str:
        """
        Joins the authors' names the way a copyright line names its holders.
        """
        return join_names([person.name for person in self.authors])


class Person(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One person `CITATION.cff` credits, as the Citation File Format writes
    a person.
    """

    family: str = Field(validation_alias="family-names")
    """
    The person's family names.
    """

    given: str = Field(validation_alias="given-names")
    """
    The person's given names.
    """

    @property
    def name(self) -> str:
        """
        Joins the given names and the family names, the order a copyright
        line gives them in.
        """
        return f"{self.given} {self.family}"


class Parity(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One declaration a file restates beside the declaration it restates, each
    named the way a finding reads it.
    """

    copied: str | None
    """
    The value the restated declaration reads, or `None` where the file
    carries none.
    """

    file: Path
    """
    The file holding the restated declaration.
    """

    label: str
    """
    The restated declaration, named as the opening of a sentence.
    """

    origin: str
    """
    The declaration restated, named as the close of a sentence.
    """

    original: str
    """
    The value the declaration restated reads.
    """

    @property
    def finding(self) -> Finding | None:
        """
        Names the restated declaration where it reads anything but the
        original, or where the file carries none.

        Returns:
            The finding, or `None` where the two agree.
        """
        if self.copied == self.original:
            return None

        reading = "is missing" if self.copied is None else f"reads `{self.copied}`"
        return Finding(
            file    = self.file,
            message = f"{self.label} {reading}, where {self.origin} reads "
            f"`{self.original}`"
        )


class Template(Document):
    """
    One issue template under `.github/ISSUE_TEMPLATE/`, either a Markdown
    file whose front matter declares it or an issue form.
    """

    labels: Annotated[list[str], BeforeValidator(read_labels)] = []
    """
    The labels an issue opened from the template takes.
    """


class Release(Document, title=".github/release.yml"):
    """
    The release-notes categories `.github/release.yml` declares.
    """

    categories: list[Category] = Field(
        validation_alias = AliasPath("changelog", "categories")
    )
    """
    The categories, in the order GitHub tries them, filing a pull request
    under the first whose labels it carries.
    """

    @property
    def names(self) -> list[str]:
        """
        Lists each label a category names, in the order the categories name
        them, leaving out the `*` that names no label.
        """
        return [
            name
            for category in self.categories
            for name in category.labels
            if name != "*"
        ]


class Manifest(Document, title="pyproject.toml"):
    """
    The declarations `pyproject.toml` makes that another file restates or
    that the build reads past `uv.lock`.
    """

    authors: list[Author] = Field(validation_alias=AliasPath("project", "authors"))
    """
    The authors `[project]` names.
    """

    constraints: list[Dependency] = Field(
        [],
        validation_alias = AliasPath("tool", "uv", "build-constraint-dependencies")
    )
    """
    The build constraints `[tool.uv]` holds hatchling's own dependencies to.
    """

    floor: Minor = Field(validation_alias=AliasPath("project", "requires-python"))
    """
    The major and minor `requires-python` sets as the floor.
    """

    license: str = Field(validation_alias=AliasPath("project", "license"))
    """
    The SPDX expression `[project].license` declares.
    """

    packages: list[Path] = Field(
        validation_alias = AliasPath(
            "tool",
            "hatch",
            "build",
            "targets",
            "wheel",
            "packages"
        )
    )
    """
    The folders the wheel carries, relative to the root.
    """

    readme: Path = Field(validation_alias=AliasPath("project", "readme"))
    """
    The README `[project]` names, relative to the root.
    """

    repository: str = Field(
        validation_alias = AliasPath("project", "urls", "Repository")
    )
    """
    The address of the repository `[project.urls]` names.
    """

    required_version: str = Field(
        validation_alias = AliasPath("tool", "uv", "required-version")
    )
    """
    The `uv` release `[tool.uv]` requires.
    """

    requires: list[Dependency] = Field(
        validation_alias = AliasPath("build-system", "requires")
    )
    """
    The requirements `[build-system]` builds the wheel with.
    """

    scripts: dict[str, str] = Field(
        {},
        validation_alias = AliasPath("project", "scripts")
    )
    """
    The object each script `[project.scripts]` declares targets, by script.
    """

    target_version: str = Field(
        validation_alias = AliasPath("tool", "prose", "target-version")
    )
    """
    The Python version `[tool.prose]` formats for.
    """

    version: str = Field(validation_alias=AliasPath("project", "version"))
    """
    The release `[project]` carries, the version a tag names.
    """

    @property
    def holders(self) -> str:
        """
        Joins the authors' names the way a copyright line names its holders.
        """
        return join_names([author.name for author in self.authors])


class Task(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One task the checkout declares, as `mise tasks ls --json` lists it. A
    file task names the file it lives in, whereas a task declared in TOML
    leaves `file` unset and carries the lines it runs.
    """

    file: Path | None
    """
    The file holding a file task, as mise resolves it.
    """

    run: Annotated[tuple[str, ...], BeforeValidator(read_commands)]
    """
    The lines a task declared in TOML runs in a shell.
    """

    source: Path
    """
    The file declaring the task, as mise resolves it.
    """


def join_names(names: Sequence[str]) -> str:
    """
    Joins `names` into one English list, the way a copyright line names
    its holders, two by `and` and three or more by commas with `and` before
    the last.
    """
    if len(names) < 3:
        return " and ".join(names)

    return f"{', '.join(names[:-1])}, and {names[-1]}"
