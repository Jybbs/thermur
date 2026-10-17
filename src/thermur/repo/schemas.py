"""
Holds the records the `repo` tasks read and report:

- The `Document` base and the files it validates, meaning the `Manifest`
  with its `Author` rows, the `Config`, the `GithubSettings` with its
  `Actions`, `Repository`, `Toggles`, and `Token` tables and the `Analysis`
  of `Feature` rows its `Repository` holds, the `Labels` with their `Label`
  rows, and the `Lock` out of the TOML, each `Workflow` with its `Job`,
  `Step`, and `Trigger` rows, each composite `Action`, the `Release` with
  its `Category` rows, each issue `Template`, and the `Citation` with its
  `Person` rows out of the YAML, each `Ruleset` with its `Rule` and `Actor`
  rows and the `Parameters` of `Status` rows each rule takes out of the JSON
  under `.github/rulesets/`, the site's `Package` out of `docs/package.json`,
  and the `Guide` out of the contributor guide's tables
- The `Policy`, `Scope`, `SquashMessage`, `SquashTitle`, and `State` enums
  each closed-set value in `.github/settings.toml` takes, and the
  `ActorType`, `BypassMode`, `Enforcement`, `MergeMethod`, `RuleType`, and
  `Target` enums each closed-set value of a ruleset takes
- Each `Request` `repo:rulesets` sends GitHub
- Each `Task` mise lists and each `Issue` mise reports on one
- Each `Finding` a check makes at its `Level`, and each `Parity` a check
  compares
- Each `Lag` `lock:lift` reports with the `Edit` rows that lift it, the
  `Audit` of `Vulnerability` rows `uv audit` reports, and the `Tree` of
  `Resolution` rows `uv tree` resolves
"""

from collections     import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime        import datetime
from enum            import StrEnum, auto
from http            import HTTPMethod
from itertools       import cycle
from json            import loads as from_json
from operator        import attrgetter
from packaging.requirements import Requirement
from packaging.specifiers   import Specifier
from packaging.utils        import canonicalize_name
from packaging.version      import Version
from pathlib                import Path, PurePosixPath
from pydantic               import (
    AfterValidator,
    AliasPath,
    BaseModel,
    BeforeValidator,
    Field,
    JsonValue,
    PlainValidator,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator
)
from re           import MULTILINE, escape, findall, fullmatch, match, search, sub, subn
from tomllib      import loads
from typing       import Annotated, Self
from urllib.parse import urlsplit
from yaml_rs      import loads as from_yaml

type Dependency = Annotated[Requirement, PlainValidator(Requirement)]
type Major      = Annotated[str, AfterValidator(read_major)]
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


class ActorType(StrEnum):
    """
    The kind of actor a ruleset lets bypass it, as `actor_type` names it.
    """

    DEPLOY_KEY         = "DeployKey"
    INTEGRATION        = "Integration"
    ORGANIZATION_ADMIN = "OrganizationAdmin"
    REPOSITORY_ROLE    = "RepositoryRole"
    TEAM               = "Team"
    USER               = "User"


def read_major(version: str) -> str:
    """
    Reads the major in `version`, the part ahead of its first dot, so
    `24.11.1` reads as `24`.
    """
    return version.partition(".")[0]


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


def read_pairs[V](pairs: list[tuple[str, V]]) -> dict[str, V]:
    """
    Builds one mapping out of key and value pairs, such as the rows of a
    table or the members of a JSON object.

    Raises:
        ValueError: Naming each key two or more pairs share, which
                    validation reports against the field or the document.
    """
    if repeats := [f"`{key}`" for key in repeated(key for key, _ in pairs)]:
        raise ValueError(f"names {join_names(repeats)} more than once")

    return dict(pairs)


def repeated[T](values: Iterable[T]) -> list[T]:
    """
    Keeps each of `values` that appears more than once, in the order of its
    first appearance.
    """
    return [value for value, count in Counter(values).items() if count > 1]


def read_versions(packages: Iterable[dict]) -> dict[str, str]:
    """
    Maps each package `uv.lock` resolves to the version it resolves.
    """
    return {package["name"]: package["version"] for package in packages}


class Author(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One author `[project].authors` names.
    """

    name: str
    """
    The author's name as a copyright line gives it.
    """


class Document(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    A file of the checkout, validated from its text into the fields the
    `repo` tasks read, whose validation error carries the file's path from
    the root as its title.
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
        TOML where the suffix is `.toml` or `.lock`, as JSON where it is
        `.json`, as the YAML front matter opening a Markdown file where it
        is `.md`, and as YAML otherwise, keeping a YAML timestamp as its
        text.

        Raises:
            ValueError: Where the TOML, the JSON, or the YAML fails to
                        parse, where a JSON object names a key more than
                        once, or where a Markdown file opens on no front
                        matter, which validation reports against the whole
                        document.
        """
        if suffix in (".lock", ".toml"):
            return loads(text)

        if suffix == ".json":
            return from_json(text, object_pairs_hook=read_pairs)

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


class BypassMode(StrEnum):
    """
    When an actor may bypass a ruleset, as `bypass_mode` names it.
    """

    ALWAYS       = auto()
    EXEMPT       = auto()
    PULL_REQUEST = auto()


class Config(Document, title=".mise/config.toml"):
    """
    The tools `.mise/config.toml` pins, the floor it sets on mise itself,
    and the folder its `[env]` puts on the path.
    """

    bin: Path = Field(validation_alias=AliasPath("env", "_", "path"))
    """
    The folder `_.path` puts on the path, relative to the root.
    """

    floor: str = Field(validation_alias=AliasPath("min_version"))
    """
    The oldest release of mise that `min_version` lets read the file.
    """

    node: Major = Field(validation_alias=AliasPath("tools", "node"))
    """
    The major of the `node` release mise pins.
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

    description: str = Field(validation_alias=AliasPath("project", "description"))
    """
    The sentence `[project]` describes the project in, which `repo:rulesets`
    sends as the repository's description.
    """

    floor: Minor = Field(validation_alias=AliasPath("project", "requires-python"))
    """
    The major and minor `requires-python` sets as the floor.
    """

    homepage: str = Field(validation_alias=AliasPath("project", "urls", "Homepage"))
    """
    The address of the docs site `[project.urls]` names, with its scheme and
    its closing slash.
    """

    keywords: list[str] = Field(validation_alias=AliasPath("project", "keywords"))
    """
    The keywords `[project]` lists, which `repo:rulesets` sends as the
    repository's topics.
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
    def address(self) -> str:
        """
        Writes the homepage the way the repository's homepage field shows
        it, with its scheme and its closing slash dropped.
        """
        homepage = urlsplit(self.homepage)
        return f"{homepage.netloc}{homepage.path}".removesuffix("/")

    @property
    def holders(self) -> str:
        """
        Joins the authors' names the way a copyright line names its holders.
        """
        return join_names([author.name for author in self.authors])

    @property
    def requests(self) -> list[Request]:
        """
        Writes the calls sending the description `[project]` declares as
        the repository's description and its homepage as the repository's
        homepage, then the keywords it lists as the repository's topics,
        which replace every topic the repository carries.
        """
        return [
            Request(
                body     = {"description": self.description, "homepage": self.address},
                endpoint = "",
                method   = "PATCH"
            ),
            Request(body={"names": self.keywords}, endpoint="topics", method="PUT")
        ]


class Package(Document, title="docs/package.json"):
    """
    The pins `docs/package.json` declares that restate another pin or that
    another pin restates, each an exact version, as Bun installs it.
    """

    coverage: str = Field(
        validation_alias = AliasPath("devDependencies", "@vitest/coverage-v8")
    )
    """
    The release of Vitest's coverage provider the site's suite runs under.
    """

    types: Major = Field(validation_alias=AliasPath("devDependencies", "@types/node"))
    """
    The major of the release of Node's types the site's TypeScript checks
    against.
    """

    vitest: str = Field(validation_alias=AliasPath("devDependencies", "vitest"))
    """
    The release of Vitest the site's suite runs on.
    """


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
        Writes the command running the task, `mise` and its first alias
        where it declares one and `mise run` and its name otherwise.
        """
        return f"mise {self.aliases[0]}" if self.aliases else f"mise run {self.name}"

    @property
    def dependencies(self) -> list[Requirement]:
        """
        Reads the requirements the task file's inline metadata declares, or
        none where it declares none.
        """
        return [Requirement(entry) for entry in self.metadata.get("dependencies", [])]

    @property
    def metadata(self) -> dict:
        """
        Reads the `script` block of the task file's inline metadata under
        PEP 723, or nothing where the task holds no file or the file no
        such block.
        """
        text  = self.file.read_text(encoding="utf-8") if self.file else ""
        block = search(r"(?m)^# /// script$\s(?P<body>(^#(| .*)$\s)+)^# ///$", text)
        if not block:
            return {}

        return loads(
            "".join(
                line[2:] or "\n" for line in block["body"].splitlines(keepends=True)
            )
        )

    @property
    def path(self) -> Path:
        """
        Names the file holding the task, or the file declaring it where the
        task is declared in TOML, as mise resolves it.
        """
        return self.file or self.source


class Category(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One release-notes category `.github/release.yml` declares.
    """

    labels: list[str]
    """
    The labels any one of which files a pull request under the category,
    where `*` files every pull request no earlier category takes.
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

    @property
    def on_pull_request(self) -> bool:
        """
        Tells whether the workflow runs on a pull request, which makes its
        gate a check a ruleset can require and its rows the checks `repo:ci`
        runs.
        """
        return "pull_request" in self.events

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


class Guide(Document, extra="forbid", title=".github/CONTRIBUTING.md"):
    """
    The tables `.github/CONTRIBUTING.md` holds, each empty where the guide
    carries no table under its header.
    """

    checks: Annotated[dict[str, str], BeforeValidator(read_pairs)]
    """
    The command the checks table gives each check for a laptop, by the
    name of the check.
    """

    labels: Annotated[dict[str, str], BeforeValidator(read_pairs)]
    """
    The description the labels table gives each label, by the label's
    name.
    """

    @staticmethod
    def load(suffix: str, text: str) -> dict[str, list[tuple[str, str]]]:
        """
        Reads the rows of the checks table and of the labels table out of
        the guide's text, whatever its suffix.
        """
        return {
            "checks" : read_table("**Check**", text),
            "labels" : read_table("**Label**", text)
        }


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
            for value in repeated(map(attrgetter(field), rows))
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


class Enforcement(StrEnum):
    """
    Whether a ruleset is enforced, as `enforcement` names it.
    """

    ACTIVE   = auto()
    DISABLED = auto()
    EVALUATE = auto()


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


class Level(StrEnum):
    """
    How a finding bears on the audit, each member's value the GitHub Actions
    workflow command that prints it.
    """

    ERROR   = auto()
    WARNING = auto()


class MergeMethod(StrEnum):
    """
    A way a pull request may merge, as `allowed_merge_methods` names it.
    """

    MERGE  = auto()
    REBASE = auto()
    SQUASH = auto()


class Actor(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One actor who may bypass a ruleset.
    """

    actor_type: ActorType
    """
    The kind of actor.
    """

    actor_id: int | None = None
    """
    The id of the actor, which the rulesets API requires for an
    `Integration`, `RepositoryRole`, `Team`, or `User`, ignores for an
    `OrganizationAdmin`, and takes as `None` for a `DeployKey`.
    """

    bypass_mode: BypassMode = BypassMode.ALWAYS
    """
    When the actor may bypass the ruleset, which the rulesets API takes as
    `always` where the file sets none.
    """


class Actions(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The GitHub Actions permissions `[actions]` declares, by the field names
    the `actions/permissions` endpoint reads.
    """

    allowed_actions: Policy
    """
    The actions a workflow may run, `all` letting it run any.
    """

    enabled: bool
    """
    Whether GitHub Actions runs on the repository.
    """

    sha_pinning_required: bool
    """
    Whether a workflow must pin each action from another repository to a
    full commit SHA.
    """

    @property
    def request(self) -> Request:
        """
        Writes the call setting the GitHub Actions permissions.
        """
        return Request(
            body     = self.model_dump(),
            endpoint = "actions/permissions",
            method   = "PUT"
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


class Analysis(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The security and analysis features `.github/settings.toml` sets, by the
    name GitHub's API gives each.
    """

    secret_scanning: Feature
    """
    Secret scanning, which raises an alert on a credential pushed to the
    repository.
    """

    secret_scanning_push_protection: Feature
    """
    Push protection, which blocks a push carrying a credential secret
    scanning recognizes.
    """


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


class Feature(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One feature `security_and_analysis` turns on or off.
    """

    status: State
    """
    Whether the feature runs.
    """


class GithubSettings(Document, extra="forbid", title=".github/settings.toml"):
    """
    Every repository setting no ruleset carries, as `.github/settings.toml`
    declares it, holding nothing beside its tables.
    """

    actions: Actions
    """
    The GitHub Actions permissions.
    """

    repository: Repository
    """
    The repository's own settings.
    """

    toggles: Toggles
    """
    The settings GitHub serves each as an endpoint of its own.
    """

    token: Token
    """
    The defaults of the token each workflow run receives.
    """

    @property
    def requests(self) -> list[Request]:
        """
        Writes the call sending each table, the repository's own settings
        first, then the Actions permissions, the token's defaults, and each
        toggle.
        """
        return [
            self.repository.request, self.actions.request, self.token.request,
            *self.toggles.requests
        ]


def read_table(header: str, text: str) -> list[tuple[str, str]]:
    """
    Reads the first two cells of each row of the Markdown table in `text`
    whose header row opens on the cell `header`, reading a cell filled by
    one code span without its backticks, the table running from the row
    under its divider to the first blank line.
    """
    body = text.partition(f"\n| {header} |")[2].partition("\n\n")[0]
    return [
        tuple(sub(r"^`([^`]*)`$", r"\1", cell) for cell in row)
        for row in findall(r"(?m)^\| (.+?) \| (.+?) \|", body)
    ]


class Parameters(BaseModel, extra="allow", frozen=True, use_attribute_docstrings=True):
    """
    The parameters one rule of a ruleset takes, validated for the
    merge methods a `pull_request` rule allows and the checks a
    `required_status_checks` rule requires, and kept as written otherwise.
    """

    allowed_merge_methods: tuple[MergeMethod, ...] = ()
    """
    The merge methods the rule allows, or none for a rule of another type.
    """

    required_status_checks: tuple[Status, ...] = ()
    """
    The checks the rule requires, or none for a rule of another type.
    """


class Policy(StrEnum):
    """
    The actions a workflow may run, as `allowed_actions` names them.
    """

    ALL        = auto()
    LOCAL_ONLY = auto()
    SELECTED   = auto()


class Repository(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The repository's own settings `[repository]` declares, by the field
    names `PATCH /repos/{owner}/{repo}` reads.
    """

    allow_auto_merge: bool
    """
    Whether a pull request can be set to merge once its checks pass.
    """

    allow_merge_commit: bool
    """
    Whether a pull request can merge as a merge commit.
    """

    allow_rebase_merge: bool
    """
    Whether a pull request can merge by rebasing its commits.
    """

    allow_squash_merge: bool
    """
    Whether a pull request can merge as one squashed commit.
    """

    allow_update_branch: bool
    """
    Whether a pull request offers to take in its base branch once it falls
    behind.
    """

    delete_branch_on_merge: bool
    """
    Whether a pull request's head branch is deleted once it merges.
    """

    has_discussions: bool
    """
    Whether the repository shows the Discussions tab.
    """

    has_issues: bool
    """
    Whether the repository takes issues.
    """

    has_projects: bool
    """
    Whether the repository shows the Projects tab.
    """

    has_wiki: bool
    """
    Whether the repository shows the wiki.
    """

    security_and_analysis: Analysis
    """
    The security and analysis features the repository runs.
    """

    squash_merge_commit_message: SquashMessage
    """
    The message a squash merge's commit takes by default.
    """

    squash_merge_commit_title: SquashTitle
    """
    The title a squash merge's commit takes by default.
    """

    @property
    def request(self) -> Request:
        """
        Writes the call setting the repository's own settings.
        """
        return Request(body=self.model_dump(), endpoint="", method="PATCH")


class Request(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One call `repo:rulesets` makes to GitHub's REST API through `gh api`, on
    the repository the checkout's remote names, which `gh` fills in for the
    `{owner}` and `{repo}` placeholders.
    """

    endpoint: str
    """
    The path of the call under `repos/{owner}/{repo}`, empty for the
    repository itself.
    """

    method: HTTPMethod
    """
    The HTTP method the call takes.
    """

    body: dict[str, JsonValue] | None = None
    """
    The JSON body the call sends, or `None` where it sends none.
    """

    @property
    def command(self) -> list[str]:
        """
        Writes the `gh api` command making the call, which reads the body
        from its standard input where the call sends one and prints no
        response.
        """
        return [
            "gh", "api",
            *(["--input", "-"] if self.body is not None else []),
            "--method", self.method, "--silent",
            str(PurePosixPath("repos/{owner}/{repo}", self.endpoint))
        ]

    @property
    def payload(self) -> bytes | None:
        """
        Encodes the body as the JSON `command` reads from its standard
        input, or `None` where the call sends no body.
        """
        return None if self.body is None else TypeAdapter(JsonValue).dump_json(
            self.body
        )


class Rule(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One rule a ruleset holds its refs to.
    """

    type: RuleType
    """
    The kind of rule, as the rulesets API names it.
    """

    parameters: Parameters = Field(default_factory=Parameters)
    """
    The parameters the rule takes, or none for a rule taking none.
    """


class RuleType(StrEnum):
    """
    The kind of rule a ruleset holds, as each rule's `type` names it.
    """

    BRANCH_NAME_PATTERN         = auto()
    CODE_COVERAGE               = auto()
    CODE_QUALITY                = auto()
    CODE_SCANNING               = auto()
    COMMIT_AUTHOR_EMAIL_PATTERN = auto()
    COMMIT_MESSAGE_PATTERN      = auto()
    COMMITTER_EMAIL_PATTERN     = auto()
    COPILOT_CODE_REVIEW         = auto()
    CREATION = auto()
    DELETION = auto()
    FILE_EXTENSION_RESTRICTION  = auto()
    FILE_PATH_RESTRICTION       = auto()
    LICENSE_COMPLIANCE_SCANNING = auto()
    MAX_FILE_PATH_LENGTH        = auto()
    MAX_FILE_SIZE               = auto()
    MERGE_QUEUE                 = auto()
    NON_FAST_FORWARD            = auto()
    PULL_REQUEST                = auto()
    REQUIRED_DEPLOYMENTS        = auto()
    REQUIRED_LINEAR_HISTORY     = auto()
    REQUIRED_SIGNATURES         = auto()
    REQUIRED_STATUS_CHECKS      = auto()
    TAG_NAME_PATTERN            = auto()
    UPDATE    = auto()
    WORKFLOWS = auto()


class Ruleset(Document, extra="forbid"):
    """
    One ruleset under `.github/rulesets/`, holding the fields the rulesets
    API takes for creating or updating a ruleset on a repository.
    """

    bypass_actors: tuple[Actor, ...]
    """
    The actors who may bypass the ruleset.
    """

    conditions: dict[str, JsonValue]
    """
    The refs the ruleset applies to, by the patterns they match.
    """

    enforcement: Enforcement
    """
    Whether the ruleset is enforced.
    """

    name: str
    """
    The name the ruleset carries on GitHub, which `repo:rulesets` finds the
    live ruleset by.
    """

    rules: tuple[Rule, ...]
    """
    The rules the ruleset holds its refs to.
    """

    target: Target
    """
    What the ruleset applies to, meaning branches, tags, or every push.
    """

    @property
    def contexts(self) -> set[str]:
        """
        Collects the context of each check the ruleset requires to pass.
        """
        return {
            check.context
            for rule in self.rules
            for check in rule.parameters.required_status_checks
        }

    def request(self, live: int | None) -> Request:
        """
        Writes the call updating the live ruleset whose id is `live` in
        place, or creating the ruleset where `live` is `None`, sending each
        field as the file declares it.
        """
        endpoint, method = (
            ("rulesets", "POST") if live is None else (f"rulesets/{live}", "PUT")
        )
        return Request(
            body     = self.model_dump(exclude_unset=True, mode="json"),
            endpoint = endpoint,
            method   = method
        )


class Scope(StrEnum):
    """
    The scope a workflow run's token takes where the workflow sets no
    `permissions`, as `default_workflow_permissions` names it.
    """

    READ  = auto()
    WRITE = auto()


class SquashMessage(StrEnum):
    """
    The message a squash merge's commit takes by default, as
    `squash_merge_commit_message` names it.
    """

    BLANK           = "BLANK"
    COMMIT_MESSAGES = "COMMIT_MESSAGES"
    PR_BODY         = "PR_BODY"


class SquashTitle(StrEnum):
    """
    The title a squash merge's commit takes by default, as
    `squash_merge_commit_title` names it.
    """

    COMMIT_OR_PR_TITLE = "COMMIT_OR_PR_TITLE"
    PR_TITLE           = "PR_TITLE"


class State(StrEnum):
    """
    Whether a security and analysis feature runs, as its `status` names it.
    """

    DISABLED = auto()
    ENABLED  = auto()


class Status(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One check a `required_status_checks` rule requires to pass.
    """

    context: str
    """
    The name the check reports under, which for a GitHub Actions job is the
    job's `name`.
    """

    integration_id: int | None = None
    """
    The id of the app the check must come from, or `None` where any app
    may report it.
    """


class Target(StrEnum):
    """
    What a ruleset applies to, meaning branches, tags, or every push to the
    repository, as `target` names it.
    """

    BRANCH = auto()
    PUSH   = auto()
    TAG    = auto()


class Toggles(
    BaseModel,
    alias_generator          = lambda name: name.replace("_", "-"),
    extra                    = "forbid",
    frozen                   = True,
    use_attribute_docstrings = True
):
    """
    The settings GitHub serves each as an endpoint of its own under
    `/repos/{owner}/{repo}`, which `PUT` turns on and `DELETE` turns off,
    keyed under `[toggles]` by the endpoint's name.
    """

    automated_security_fixes: bool
    """
    Dependabot security updates, which open a pull request moving a
    vulnerable pin.
    """

    immutable_releases: bool
    """
    Immutable releases, which keep a published release's tag and assets
    from changing.
    """

    private_vulnerability_reporting: bool
    """
    Private vulnerability reporting, which lets a reporter send a
    vulnerability privately.
    """

    vulnerability_alerts: bool
    """
    Dependabot alerts, raised on a dependency carrying a published
    vulnerability.
    """

    @property
    def requests(self) -> list[Request]:
        """
        Writes the call turning each setting on or off, in the order the
        record declares its fields.
        """
        return [
            Request(endpoint=endpoint, method="PUT" if on else "DELETE")
            for endpoint, on in self.model_dump(by_alias=True).items()
        ]


class Edit(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One place a file declares a pin, found as the release it names behind
    text a pattern matches, which `lock:lift` rewrites to the newest
    release.
    """

    ahead: str
    """
    The pattern the text ahead of the release matches.
    """

    current: str
    """
    The release the file names.
    """

    file: Path
    """
    The file declaring the pin, relative to the root.
    """

    newest: str
    """
    The release the lift writes in place of `current`.
    """

    def write(self, root: Path):
        """
        Writes `newest` in place of each `current` the file under `root`
        names behind text `ahead` matches, where no further digit of a
        release follows it. `^` and `$` in `ahead` match at each line.

        Raises:
            ValueError: Where the file names `current` nowhere behind such
                        text.
        """
        path        = root / self.file
        text, count = subn(
            rf"({self.ahead}){escape(self.current)}(?!\.?\d)",
            lambda found: found[1] + self.newest,
            path.read_text(encoding="utf-8"),
            flags = MULTILINE
        )
        if not count:
            raise ValueError(f"`{self.file}` names no `{self.current}` to lift")

        path.write_text(text, encoding="utf-8")


class Resolution(
    BaseModel,
    extra="ignore",
    frozen=True,
    use_attribute_docstrings=True
):
    """
    One package `uv tree --format json` lists, at the release uv resolves
    it to.
    """

    version: str
    """
    The release uv resolves.
    """

    latest: str | None = Field(None, validation_alias="latest_version")
    """
    The newest release on the index, which `uv tree --outdated` reports, or
    `None` where it is not asked.
    """

    uploads: list[datetime] = []
    """
    When each file of the release was uploaded.
    """

    @property
    def uploaded(self) -> datetime | None:
        """
        Reads when the release's first file was uploaded, or `None` where uv
        lists no upload time.
        """
        return min(self.uploads, default=None)

    @model_validator(mode="before")
    @classmethod
    def gather(cls, package: dict) -> dict:
        """
        Gathers the upload time uv lists for the source distribution and for
        each wheel of the release under `uploads`.
        """
        return package | {
            "uploads": [
                file["upload_time"]
                for file in [*package.get("wheels", []), package.get("sdist") or {}]
                if file.get("upload_time")
            ]
        }


def read_resolutions(resolution: dict[str, dict]) -> dict[str, dict]:
    """
    Keeps each package `uv tree --format json` lists under `resolution`, by
    its normalized name, leaving out the script the tree is rooted at.
    """
    return {
        canonicalize_name(entry["name"]): entry
        for entry in resolution.values()
        if entry["kind"] == "package"
    }


class Token(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The defaults of the token each workflow run receives, which `[token]`
    declares by the field names the `actions/permissions/workflow` endpoint
    reads.
    """

    can_approve_pull_request_reviews: bool
    """
    Whether a workflow can approve a pull request through the token.
    """

    default_workflow_permissions: Scope
    """
    The scope the token takes where a workflow sets no `permissions`.
    """

    @property
    def request(self) -> Request:
        """
        Writes the call setting the defaults of the token.
        """
        return Request(
            body     = self.model_dump(),
            endpoint = "actions/permissions/workflow",
            method   = "PUT"
        )


class Vulnerability(
    BaseModel,
    extra="ignore",
    frozen=True,
    use_attribute_docstrings=True
):
    """
    One advisory `uv audit --output-format json` reports on a package a
    lockfile resolves.
    """

    id: str
    """
    The advisory's identifier.
    """

    fixes: tuple[str, ...] = Field((), validation_alias="fix_versions")
    """
    Each release that fixes the advisory, one for each affected line.
    """

    name: str = Field(validation_alias=AliasPath("dependency", "name"))
    """
    The package the advisory concerns.
    """

    version: str = Field(validation_alias=AliasPath("dependency", "version"))
    """
    The release of the package the lockfile resolves.
    """


class Audit(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    The advisories `uv audit --output-format json` reports on one lockfile.
    """

    vulnerabilities: tuple[Vulnerability, ...] = ()
    """
    Each advisory, in the order uv reports them.
    """

    def clears(self, current: str, name: str, newest: str) -> tuple[str, ...]:
        """
        Keeps each advisory on the package `name` at `current` that a
        release above `current` and no later than `newest` fixes.
        """
        return tuple(
            vulnerability.id
            for vulnerability in self.vulnerabilities
            if canonicalize_name(vulnerability.name) == canonicalize_name(name)
            and vulnerability.version                == current
            and any(
                Version(current) < Version(fix) <= Version(newest)
                for fix in vulnerability.fixes
            )
        )


class Lag(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One pin lagging its newest release, as one row of the table `lock:lift`
    prints.
    """

    current: str
    """
    The release the checkout pins.
    """

    newest: str
    """
    The newest release the pin's manager reports.
    """

    pin: str
    """
    The name the manager gives the pin, which `--hold` names.
    """

    advisories: tuple[str, ...] = ()
    """
    The identifier of each advisory the lift clears.
    """

    edits: tuple[Edit, ...] = ()
    """
    Each place a file declares the pin, where the lift writes the newest
    release.
    """

    published: datetime | None = None
    """
    When the newest release was published, or `None` where its manager
    reports no date.
    """

    upgraded: bool = False
    """
    Whether the pin is a package `uv.lock` resolves, which the lift moves
    through `uv lock --upgrade-package`.
    """

    @property
    def files(self) -> list[Path]:
        """
        Lists each file declaring the pin, in the order of its edits, then
        `uv.lock` for a package it resolves.
        """
        return [
            *dict.fromkeys(edit.file for edit in self.edits),
            *[Path(Lock.model_config["title"])] * self.upgraded
        ]

    @property
    def major(self) -> bool:
        """
        Tells whether the lift changes the left-most non-zero part of the
        release, so `0.33.1` to `0.34.0` does and `2026.7.0` to `2026.9.0`
        does not.
        """
        current = read_release(self.current)
        place   = next(
            (index for index, part in enumerate(current) if part),
            len(current)
        )
        return read_release(self.newest)[: place + 1] != current[: place + 1]

    @classmethod
    def declared(
        cls,
        places     : Mapping[Path, str],
        *,
        current    : str,
        newest     : str,
        pin        : str,
        advisories : tuple[str, ...]       = (),
        published  : datetime | str | None = None
    ) -> Self:
        """
        Builds the lag of a pin each file `places` names declares behind
        text the pattern beside it matches, each edit writing `newest` in
        place of `current`.
        """
        return cls(
            advisories = advisories,
            current    = current,
            edits      = tuple(
                Edit(
                    ahead   = ahead,
                    current = current,
                    file    = file,
                    newest  = newest
                )
                for file, ahead in places.items()
            ),
            newest    = newest,
            pin       = pin,
            published = published
        )

    def row(self, now: datetime) -> str:
        """
        Writes the pin as a row of the table, its newest release's age
        counted in whole days up to `now`.
        """
        age = (now - self.published).days if self.published else ""
        return (
            f"| `{self.pin}` | {', '.join(f'`{file}`' for file in self.files)} | "
            f"{self.current} | {self.newest} | {'yes' if self.major else 'no'} | "
            f"{age} | {', '.join(self.advisories)} |"
        )

    def write(self, root: Path):
        """
        Writes the newest release wherever each of the pin's edits finds it
        in the files under `root`.

        Raises:
            ValueError: Where a file names no release an edit lifts.
        """
        for edit in self.edits:
            edit.write(root)


def read_pin(requirement: Requirement) -> str | None:
    """
    Reads the one release `requirement` pins through a single `==` specifier
    carrying no `.*` wildcard, or `None` where it names a range.
    """
    match list(requirement.specifier):
        case [Specifier(operator="==", version=version)] if not version.endswith(".*"):
            return version

    return None


def read_release(version: str) -> tuple[int, ...]:
    """
    Reads the dotted integers `version` opens on past a leading `v`,
    which a semantic, a PEP 440, and a calendar version each open on, so
    `v2.0.0-alpha.20` reads as `(2, 0, 0)`.
    """
    return tuple(map(int, findall(r"\d+", match(r"v?[\d.]*", version)[0])))


class Tree(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    The packages `uv tree --format json` resolves for a script.
    """

    packages: Annotated[dict[str, Resolution], BeforeValidator(read_resolutions)] = (
        Field(validation_alias="resolution")
    )
    """
    Each package, by its normalized name.
    """

    def package(self, name: str) -> Resolution | None:
        """
        Finds the package `name` names under any spelling its normalized
        name allows, or `None` where the tree holds no such package.
        """
        return self.packages.get(canonicalize_name(name))
