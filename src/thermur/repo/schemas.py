"""
Holds the records the repository audit reads and reports:

- The `Document` base and the files it validates, meaning the `Manifest`
  with its `Author` rows, the `Config`, the `Labels` with their `Label`
  rows, and the `Lock` out of the TOML, and each `Workflow` with its `Job`,
  `Step`, and `Trigger` rows, each composite `Action`, the `Release` with
  its `Category` rows, each issue `Template`, and the `Citation` with its
  `Person` rows out of the YAML
- The `Guide` with the tables the contributor guide holds
- Each `Task` mise lists and each `Issue` mise reports on one
- Each `Finding` a check makes at its `Level`, and each `Parity` a check
  compares
"""

from collections     import Counter
from collections.abc import Iterable, Sequence
from enum            import StrEnum, auto
from itertools       import cycle
from operator        import attrgetter
from packaging.requirements import Requirement
from pathlib                import Path, PurePosixPath
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
from re      import findall, fullmatch, search, sub
from tomllib import loads
from typing  import Annotated, ClassVar, Self
from yaml_rs import loads as from_yaml

type Dependency = Annotated[Requirement, PlainValidator(Requirement)]
type Minor      = Annotated[str, AfterValidator(read_minor)]
type Scalar     = bool | float | int | str


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


class Author(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One author `[project].authors` names.
    """

    name: str
    """
    The author's name as a copyright line gives it.
    """


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


def read_names(needs: str | list[str]) -> list[str]:
    """
    Reads the jobs a job waits for, which a workflow writes as one name or
    as a list of them.
    """
    return [needs] if isinstance(needs, str) else needs


def read_versions(packages: Iterable[dict]) -> dict[str, str]:
    """
    Maps each package `uv.lock` resolves to the version it resolves.
    """
    return {package["name"]: package["version"] for package in packages}


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
    A TOML or YAML file of the checkout, validated from its text into the
    fields the checks read, whose validation error carries the file's path
    from the root as its title.
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
        TOML where the suffix is `.toml` or `.lock`, as the YAML front
        matter opening a Markdown file where it is `.md`, and as YAML
        otherwise, keeping a YAML timestamp as its text.

        Raises:
            ValueError: Where the TOML or the YAML fails to parse, or where
                        a Markdown file opens on no front matter, which
                        validation reports against the whole document.
        """
        if suffix in (".lock", ".toml"):
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


class Guide(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The tables `.github/CONTRIBUTING.md` holds, each empty where the guide
    carries no table under its header.
    """

    file: ClassVar[Path] = Path(".github/CONTRIBUTING.md")

    checks: dict[str, str]
    """
    The command the checks table gives each check for a laptop, by the
    name of the check.
    """

    labels: dict[str, str]
    """
    The description the labels table gives each label, by the label's
    name.
    """

    @classmethod
    def read(cls, root: Path) -> Self:
        """
        Reads the checks table and the labels table out of the guide under
        `root`.
        """
        text = (root / cls.file).read_text(encoding="utf-8")
        return cls(
            checks = read_table("**Check**", text),
            labels = read_table("**Label**", text)
        )


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


class Label(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One label `.github/labels.toml` declares, as GitHub shows it on an issue
    or a pull request.
    """

    color: Annotated[str, Field(pattern=r"^[0-9a-f]{6}$")]
    """
    The label's color as six lowercase hex digits.
    """

    description: Annotated[str, Field(max_length=100)]
    """
    What the label covers, in one line of at most the 100 characters
    GitHub's label API accepts.
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


class Level(StrEnum):
    """
    How a finding bears on the audit, each member's value the GitHub Actions
    workflow command that prints it.
    """

    ERROR   = auto()
    WARNING = auto()


class Lock(Document, title="uv.lock"):
    """
    The version `uv.lock` resolves for each package.
    """

    versions: Annotated[dict[str, str], BeforeValidator(read_versions)] = Field(
        validation_alias = AliasPath("package")
    )
    """
    The version each package resolves to, by its normalized name.
    """


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


class Step(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One step a workflow job or a composite action runs, which either names
    an action or runs shell lines.
    """

    env: dict[str, Scalar] = {}
    """
    The variables the step sets for what it runs, by name.
    """

    inputs: dict[str, Scalar] = Field({}, validation_alias="with")
    """
    The inputs the step passes the action it names, by input.
    """

    run: str = ""
    """
    The shell lines the step runs, or nothing where it names an action.
    """

    uses: str = ""
    """
    The action the step runs, or nothing where it runs shell lines.
    """

    @property
    def archive(self) -> str | None:
        """
        Reads the `cache_key` the step passes up to the `-` before the
        `hashFiles` call ending it, which names the archive every save of
        one tool subset shares.

        Returns:
            The key up to that `-`, or `None` where the step passes no
            `cache_key` ending on a `-` and a `hashFiles` call.
        """
        key = fullmatch(
            r"(.*)-\$\{\{\s*hashFiles\([^)]*\)\s*\}\}",
            str(self.inputs.get("cache_key", ""))
        )
        return key[1] if key else None

    @property
    def pin(self) -> tuple[str, str] | None:
        """
        Splits `uses` into the action from another repository and the commit
        it is pinned to.

        Returns:
            The action and the commit, or `None` where the step names no
            action pinned with `@`.
        """
        action, at, commit = self.uses.partition("@")
        return (action, commit) if at else None

    @property
    def saves(self) -> bool:
        """
        Tells whether the step passes the action it names a `save` reading
        `true`, quoted or not, which saves the archive the action installs.
        """
        return str(self.inputs.get("save")).lower() == "true"

    @property
    def summarizes(self) -> bool:
        """
        Tells whether the step writes to the workflow step summary, which it
        does by naming the variable holding the summary's path or by running
        `gha:brief`, which reads that variable.
        """
        return search(
            r"GITHUB_STEP_SUMMARY|gha:brief",
            self.model_dump_json()
        ) is not None


class Trigger(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One event a workflow runs on, narrowed to the files a change touches.
    """

    paths: tuple[str, ...] = ()
    """
    The globs naming the files whose change runs the workflow on this
    event, or none where any change runs it.
    """

    def matches(self, file: str) -> bool:
        """
        Tells whether a change to `file`, relative to the root, runs the
        workflow, which it does under any glob matching the file's whole
        path and under no globs at all.
        """
        return not self.paths or any(
            PurePosixPath(file).full_match(glob) for glob in self.paths
        )


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

    name: str
    """
    The name a reader runs the task by.
    """

    run: Annotated[tuple[str, ...], BeforeValidator(read_commands)]
    """
    The lines a task declared in TOML runs in a shell.
    """

    source: Path
    """
    The file declaring the task, as mise resolves it.
    """

    aliases: tuple[str, ...] = ()
    """
    The other names a reader runs the task by, in the order the task
    declares them.
    """

    @property
    def command(self) -> str:
        """
        Writes the shortest command running the task, `mise` and its first
        alias where it declares one and `mise run` and its name otherwise.
        """
        return f"mise {self.aliases[0]}" if self.aliases else f"mise run {self.name}"

    @property
    def dependencies(self) -> list[Requirement]:
        """
        Reads the requirements the `script` block of the task file's inline
        metadata declares under PEP 723, or none where the task holds no
        file or the file no such block.
        """
        text  = self.file.read_text(encoding="utf-8") if self.file else ""
        block = search(r"(?m)^# /// script$\s(?P<body>(^#(| .*)$\s)+)^# ///$", text)
        if not block:
            return []

        metadata = loads(
            "".join(
                line[2:] or "\n" for line in block["body"].splitlines(keepends=True)
            )
        )
        return [Requirement(entry) for entry in metadata.get("dependencies", [])]

    @property
    def path(self) -> Path:
        """
        Names the file holding the task, or the file declaring it where the
        task is declared in TOML, as mise resolves it.
        """
        return self.file or self.source


class Action(Document):
    """
    One composite action under `.github/actions/`, with the inputs its
    manifest declares and the steps it runs.
    """

    inputs: dict[str, dict[str, Scalar]] = {}
    """
    The inputs the manifest declares, by input, each with its default and its
    description.
    """

    steps: tuple[Step, ...] = Field(validation_alias=AliasPath("runs", "steps"))
    """
    The steps the action runs, in order.
    """


class Job(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One job a workflow runs, with the rows its matrix includes.
    """

    condition: str = Field("", validation_alias="if")
    """
    The expression deciding whether the job runs, or nothing where it runs
    once every job it waits for succeeds.
    """

    image: str = Field(validation_alias="runs-on")
    """
    The runner image the job runs on.
    """

    name: str = ""
    """
    The name the workflow gives the job, where a `${{ matrix.<key> }}`
    expression names each row's check run for that row's value under
    `key`, or nothing where the run takes the job's key, followed on a
    matrix job by each row's values in parentheses.
    """

    needs: Annotated[tuple[str, ...], BeforeValidator(read_names)] = ()
    """
    The jobs this one waits for, by key.
    """

    rows: tuple[dict[str, str], ...] = Field(
        (),
        validation_alias = AliasPath("strategy", "matrix", "include")
    )
    """
    The rows the job's matrix includes, each running the job once.
    """

    steps: tuple[Step, ...] = ()
    """
    The steps the job runs, in order.
    """

    @property
    def always(self) -> bool:
        """
        Tells whether the job runs under `always()`, with or without the
        `${{ }}` GitHub lets a condition leave out, so it runs whatever the
        jobs it waits for end in.
        """
        return (
            (
                self.condition.removeprefix("${{")
                    .removesuffix("}}")
                    .strip()
            ) == "always()"
        )

    @property
    def checks(self) -> list[tuple[str, str]]:
        """
        Pairs the name each row's check run takes with the task the row
        runs, for a row naming one under `task`, reading the name through a
        `${{ matrix.<key> }}` expression in the job's `name` and as written
        for every row where `name` holds none.
        """
        return [
            (name, row["task"])
            for name, row in zip(cycle(self.expand(self.name)), self.rows)
            if "task" in row
        ]

    def expand(self, value: str) -> list[str]:
        """
        Reads the values `value` takes across the job's rows:

        - For a `${{ matrix.<key> }}` expression, each row's value under
          `key`, read as an empty string where the row declares no `key`,
          as GitHub reads a property the matrix leaves out
        - For any other value, the value as written
        """
        if key := fullmatch(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", value):
            return [row.get(key[1], "") for row in self.rows]

        return [value]


class Workflow(Document):
    """
    One workflow under `.github/workflows/`, with the events it runs on and
    the jobs it runs.
    """

    jobs: dict[str, Job]
    """
    The jobs the workflow runs, by key.
    """

    name: str
    """
    The name the workflow's runs carry.
    """

    events: dict[str, Trigger | None] = Field(validation_alias="on")
    """
    The events the workflow runs on, each narrowed by its filters where it
    carries any.
    """

    @property
    def checks(self) -> list[tuple[str, str]]:
        """
        Pairs the name each row of each job gives its check run with the
        task the row runs.
        """
        return [check for job in self.jobs.values() for check in job.checks]

    @property
    def gate(self) -> Job | None:
        """
        Reads the job under the key `gate`, which the workflow ends on, or
        `None` where the workflow holds none.
        """
        return self.jobs.get("gate")

    @property
    def gated(self) -> dict[str, Job]:
        """
        Keeps the jobs other than the gate, by key, which the gate waits
        for.
        """
        return {key: job for key, job in self.jobs.items() if key != "gate"}

    def pushes(self, file: str) -> bool:
        """
        Tells whether a push changing `file`, relative to the root, runs
        the workflow.
        """
        return "push" in self.events and (self.events["push"] or Trigger()).matches(
            file
        )

    def saves(self, action: str) -> bool:
        """
        Tells whether a step of the workflow passes the composite action
        that `action` names a `save` reading `true`, which saves the archive
        it installs.
        """
        return any(
            step.uses == action and step.saves
            for job in self.jobs.values()
            for step in job.steps
        )


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

    level: Level = Level.ERROR
    """
    Whether the finding fails the audit or is printed as a warning alone.
    """

    @property
    def annotation(self) -> str:
        """
        Writes the finding as the GitHub Actions annotation of its level
        the runner shows on `file`, percent-encoding each line break and
        `%`, and each `,` and `:` in the file name, which a workflow command
        reads as its own syntax.
        """
        escapes = {"%": "%25", "\n": "%0A", "\r": "%0D"}
        place   = str(self.file).translate(
            str.maketrans(escapes | {",": "%2C", ":": "%3A"})
        )
        return (
            f"::{self.level} file={place}::"
            f"{self.message.translate(str.maketrans(escapes))}"
        )


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


class Issue(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One problem `mise tasks validate --json` reports on a task.
    """

    message: str
    """
    The problem, as mise words it.
    """

    severity: Level
    """
    Whether mise counts the problem as an error or as a warning.
    """

    task: str
    """
    The name of the task the problem sits on.
    """

    details: str = ""
    """
    What mise adds to the message, such as the tasks an alias is shared by.
    """

    def finding(self, file: Path) -> Finding:
        """
        Writes the problem as a finding at its severity on `file`, the file
        holding or declaring the task.
        """
        return Finding(
            file    = file,
            level   = self.severity,
            message = f"`mise tasks validate` reports `{self.message}` on "
            f"`{self.task}`" + (f" (`{self.details}`)" if self.details else "")
        )


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


class Template(Document):
    """
    One issue template under `.github/ISSUE_TEMPLATE/`, either a Markdown
    file whose front matter declares it or an issue form.
    """

    labels: Annotated[list[str], BeforeValidator(read_labels)] = []
    """
    The labels an issue opened from the template takes.
    """


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


def join_names(names: Sequence[str]) -> str:
    """
    Joins `names` into one English list, the way a copyright line names
    its holders, two by `and` and three or more by commas with `and` before
    the last.
    """
    if len(names) < 3:
        return " and ".join(names)

    return f"{', '.join(names[:-1])}, and {names[-1]}"


def read_table(header: str, text: str) -> dict[str, str]:
    """
    Maps the first cell of each row of the Markdown table in `text` whose
    header row opens on the cell `header` to the row's second cell, reading
    a cell filled by one code span without its backticks, the table running
    from the row under its divider to the first blank line.
    """
    body = text.partition(f"\n| {header} |")[2].partition("\n\n")[0]
    return dict(
        [sub(r"^`([^`]*)`$", r"\1", cell) for cell in row]
        for row in findall(r"(?m)^\| (.+?) \| (.+?) \|", body)
    )
