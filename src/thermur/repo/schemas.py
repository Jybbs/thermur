"""
Holds the records the repository audit and `repo:rulesets` read and report:

- The `Document` base and the files it validates, meaning the `Manifest`
  with its `Author` rows, the `Config`, the `Lock`, and the `GithubSettings` with
  their `Actions`, `Repository`, `Toggles`, and `Token` tables out of the
  TOML, each `Workflow` with its `Job`, `Step`, and `Trigger` rows and
  each composite `Action` out of the YAML under `.github/`, and each
  `Ruleset` with its `Rule` rows out of the JSON under `.github/rulesets/`
- Each `Request` `repo:rulesets` sends GitHub
- Each `Task` mise lists and each `Issue` mise reports on one
- Each `Finding` a check makes at its `Level`, and each `Parity` a check
  compares
"""

from collections.abc import Iterable
from enum            import StrEnum, auto
from http            import HTTPMethod
from json            import loads as from_json
from packaging.requirements import Requirement
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
    model_validator
)
from re      import fullmatch, search
from tomllib import loads
from typing  import Annotated, Self
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


class Actions(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The GitHub Actions permissions `[actions]` declares, by the field names
    the `actions/permissions` endpoint reads.
    """

    allowed_actions: str
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
    A TOML, YAML, or JSON file of the checkout, validated from its text into
    the fields the checks and `repo:rulesets` read, whose validation error
    carries the file's path from the root as its title.
    """

    @property
    def file(self) -> Path:
        """
        Reads the path of the file a titled model validates, relative to the
        root, out of its title.
        """
        return Path(self.model_config["title"])

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: object, info: ValidationInfo) -> object:
        """
        Reads text as TOML where the suffix the context names is `.toml` or
        `.lock`, as JSON where it is `.json`, and as YAML otherwise, keeping
        a YAML timestamp as its text, and passes a mapping already read
        through unchanged.
        """
        if not isinstance(data, str):
            return data

        if info.context["suffix"] in (".lock", ".toml"):
            return loads(data)

        if info.context["suffix"] == ".json":
            return from_json(data)

        return from_yaml(data, parse_datetime=False)

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

    @property
    def holders(self) -> str:
        """
        Joins the authors' names the way a copyright line names its holders,
        two by `and` and three or more by commas with `and` before the last.
        """
        names = [author.name for author in self.authors]
        if len(names) < 3:
            return " and ".join(names)

        return f"{', '.join(names[:-1])}, and {names[-1]}"

    @property
    def requests(self) -> list[Request]:
        """
        Writes the calls sending the description `[project]` declares
        as the repository's description and the keywords it lists as the
        repository's topics, which replace every topic the repository
        carries.
        """
        return [
            Request(
                body     = {"description": self.description},
                endpoint = "",
                method   = "PATCH"
            ),
            Request(body={"names": self.keywords}, endpoint="topics", method="PUT")
        ]


class Feature(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One feature `security_and_analysis` turns on or off.
    """

    status: str
    """
    The feature's state, `enabled` or `disabled`.
    """


class Level(StrEnum):
    """
    How a finding bears on the audit, each member's value the GitHub Actions
    workflow command that prints it.
    """

    ERROR   = auto()
    WARNING = auto()


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
    def tasks(self) -> list[str]:
        """
        Lists the task each row runs, for a row naming one under `task`.
        """
        return [row["task"] for row in self.rows if "task" in row]

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
        gate a check a ruleset can require and its rows the tasks `repo:ci`
        runs.
        """
        return "pull_request" in self.events

    @property
    def tasks(self) -> list[str]:
        """
        Lists the task each row of each job runs.
        """
        return [task for job in self.jobs.values() for task in job.tasks]

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


class Parameters(BaseModel, extra="allow", frozen=True, use_attribute_docstrings=True):
    """
    The parameters one rule of a ruleset takes, validated for the checks a
    `required_status_checks` rule requires and kept as written otherwise.
    """

    required_status_checks: tuple[Status, ...] = ()
    """
    The checks the rule requires, or none for a rule of another type.
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

    squash_merge_commit_message: str
    """
    The message a squash merge's commit takes by default.
    """

    squash_merge_commit_title: str
    """
    The title a squash merge's commit takes by default.
    """

    @property
    def request(self) -> Request:
        """
        Writes the call setting the repository's own settings.
        """
        return Request(body=self.model_dump(), endpoint="", method="PATCH")


class Rule(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One rule a ruleset holds its refs to.
    """

    type: str
    """
    The kind of rule, as the rulesets API names it.
    """

    parameters: Parameters = Parameters()
    """
    The parameters the rule takes, or none for a rule taking none.
    """


class Ruleset(Document, extra="forbid"):
    """
    One ruleset under `.github/rulesets/`, holding the fields the rulesets
    API takes for creating or updating a ruleset on a repository.
    """

    bypass_actors: tuple[dict[str, JsonValue], ...]
    """
    The actors who may bypass the ruleset.
    """

    conditions: dict[str, JsonValue]
    """
    The refs the ruleset applies to, by the patterns they match.
    """

    enforcement: str
    """
    Whether the ruleset is enforced, such as `active` or `disabled`.
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

    target: str
    """
    The kind of ref the ruleset applies to, such as `branch` or `tag`.
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

    default_workflow_permissions: str
    """
    The scope the token takes where a workflow sets no `permissions`, `read`
    or `write`.
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
