"""
Pins the records the repository audit reads and reports:

- What a document's text reads as, the key each field reads, and the path
  its error carries
- What a workflow's jobs, steps, and triggers derive, and the requirements
  a task script declares
- What a `Finding` prints at each level, the findings a `Parity` and an
  `Issue` make, and the holders a `Manifest` names
- The labels a `Template` lists, the rows a `Labels` registry accepts, the
  command a `Label` writes, and the live labels a registry omits
"""

from functools              import reduce
from packaging.requirements import Requirement
from pathlib                import Path
from pydantic               import AliasPath, ValidationError
from pytest                 import Config, mark, param, raises
from urllib.parse           import unquote

from thermur.repo.schemas import (
    Author,
    Document,
    Finding,
    Guide,
    Issue,
    Job,
    Label,
    Labels,
    Level,
    Lock,
    Manifest,
    Parity,
    Step,
    Task,
    Template,
    Trigger,
    Workflow
)

ROW      = {"color": "8c055e", "description": "Wrong output", "name": "🐞 bug"}
WORKFLOW = "name: w\non:\n  push:\njobs:\n  a:\n    runs-on: x\n"


def test_a_guide_reads_each_table_from_its_divider_to_a_blank_line(tmp_path: Path):
    """
    Asserts that the guide reads each table under the header naming it,
    from the row under its divider to the first blank line, keeping a cell
    as written unless one code span fills it, and that a guide carrying no
    table under a header reads that table as empty.
    """
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/CONTRIBUTING.md").write_text(
        "Intro.\n\n| **Label** | **Covers** |\n|---|---|\n"
        "| `🐞 bug` | Wrong output in `src` |\n| `🦜 cli` | The command line |\n"
        "\n| `📚 docs` | Past the blank line |\n"
    )

    assert Guide.read(tmp_path) == Guide(
        checks = {},
        labels = {"🐞 bug": "Wrong output in `src`", "🦜 cli": "The command line"}
    )


@mark.parametrize(
    "message",
    [
        param("line\nbreak", id="newline"),
        param("carriage\rreturn", id="carriage-return"),
        param("100%", id="percent"),
        param("a,b:c", id="separators"),
        param("100%0A", id="escape-literal")
    ]
)
def test_an_annotation_stays_on_one_line_and_reads_back(message: str):
    """
    Asserts that a message holding a line break, a `%`, a separator, or the
    literal text of an escape writes an annotation holding no line break,
    from which decoding the escapes reads the message back.
    """
    annotation = Finding(file=Path("pyproject.toml"), message=message).annotation

    assert not {"\n", "\r"} & set(annotation)
    assert unquote(annotation.removeprefix("::error file=pyproject.toml::")) == message


@mark.parametrize(
    ("file", "annotation"),
    [
        param(".mise/bin/prose", "::error file=.mise/bin/prose::m", id="plain"),
        param("a,b:c", "::error file=a%2Cb%3Ac::m", id="separators")
    ]
)
def test_an_annotation_names_its_file_for_the_runner(file: str, annotation: str):
    """
    Asserts that a finding prints as the error annotation GitHub Actions
    places on the file it names, escaping a `,` or a `:` in that name, since
    a workflow command reads either as the end of the property.
    """
    assert Finding(file=Path(file), message="m").annotation == annotation


@mark.parametrize(
    ("copied", "finding"),
    [
        param("3.14", None, id="agrees"),
        param(
            None,
            Finding(
                file    = Path("README.md"),
                message = "The badge is missing, where the manifest reads `3.14`"
            ),
            id = "missing"
        ),
        param(
            "3.13",
            Finding(
                file    = Path("README.md"),
                message = "The badge reads `3.13`, where the manifest reads `3.14`"
            ),
            id = "drifts"
        )
    ]
)
def test_a_parity_names_a_copy_that_drifts(copied: str | None, finding: Finding | None):
    """
    Asserts that a restated declaration reading another value, or missing
    from its file, makes a finding on that file naming both values, and that
    one reading the original value makes none.
    """
    parity = Parity(
        copied   = copied,
        file     = Path("README.md"),
        label    = "The badge",
        origin   = "the manifest",
        original = "3.14"
    )

    assert parity.finding == finding


@mark.parametrize(
    ("document", "field"),
    [
        param(document, name, id=f"{document.__name__}.{name}")
        for document in Document.__subclasses__()
        if "title" in document.model_config
        for name, info in document.model_fields.items()
        if info.is_required()
    ]
)
def test_each_required_field_reads_the_key_its_alias_names(
    document     : type[Document],
    field        : str,
    pytestconfig : Config
):
    """
    Asserts that removing the key a required field's alias names from the
    checkout's own file fails validation at exactly that key, so each field
    reads the TOML or YAML key its alias declares.
    """
    alias: AliasPath = document.model_fields[field].validation_alias
    *table, key      = alias.path
    file             = Path(document.model_config["title"])
    parsed           = document.load(
        file.suffix,
        (pytestconfig.rootpath / file).read_text(encoding="utf-8")
    )
    del reduce(dict.__getitem__, table, parsed)[key]

    with raises(ValidationError) as error:
        document.model_validate(parsed)

    assert [detail["loc"] for detail in error.value.errors()] == [tuple(alias.path)]


@mark.parametrize(
    ("names", "holders"),
    [
        param(["Ada"], "Ada", id="one"),
        param(["Ada", "Bea"], "Ada and Bea", id="two"),
        param(["Ada", "Bea", "Cy"], "Ada, Bea, and Cy", id="three")
    ]
)
def test_the_authors_join_the_way_a_copyright_line_names_them(
    names   : list[str],
    holders : str
):
    """
    Asserts that one author reads alone, two join by `and`, and three or
    more join by commas with `and` before the last.
    """
    manifest = Manifest.model_construct(authors=[Author(name=name) for name in names])

    assert manifest.holders == holders


@mark.parametrize(
    ("text", "location", "message"),
    [
        param("name: w\nname: v\n", (), "duplicate mapping key: 'name'", id="unread"),
        param("name: w\non:\n  push:\n", ("jobs",), "Field required", id="invalid")
    ]
)
def test_a_yaml_document_reports_its_error_under_its_path(
    location : tuple,
    message  : str,
    text     : str,
    tmp_path : Path
):
    """
    Asserts that a workflow its YAML fails to read, or that fails to
    validate, raises one error titled with the workflow's path from the
    root, located at the field where validation failed and nowhere where
    reading did.
    """
    path = Path(".github/workflows/broken.yml")
    (tmp_path / path.parent).mkdir(parents=True)
    (tmp_path / path).write_text(text)

    with raises(ValidationError) as error:
        Workflow.read(tmp_path, path)

    assert error.value.title == str(path)
    assert [(
        detail["loc"],
        message in detail["msg"]
    ) for detail in error.value.errors()] == [(location, True)]


@mark.parametrize(
    ("needs", "names"),
    [
        param("needs: check", ("check",), id="one"),
        param("needs: [a, b]", ("a", "b"), id="list"),
        param("", (), id="none")
    ]
)
def test_a_job_reads_the_jobs_it_waits_for(needs: str, names: tuple[str, ...]):
    """
    Asserts that a job waiting for one job, written as its key alone, reads
    as waiting for that job, as a job written with a list of keys does.
    """
    assert workflow(f"{WORKFLOW}    {needs}\n").jobs["a"].needs == names


@mark.parametrize(
    ("value", "values"),
    [
        param("${{ matrix.tools }}", ["python uv", "zizmor"], id="matrix"),
        param("${{matrix.tools}}", ["python uv", "zizmor"], id="unspaced"),
        param("${{ matrix.task }}", ["py:check", ""], id="undeclared"),
        param("python uv", ["python uv"], id="literal"),
        param("${{ github.token }}", ["${{ github.token }}"], id="other-context")
    ]
)
def test_a_job_expands_a_matrix_value_across_its_rows(value: str, values: list[str]):
    """
    Asserts that a `${{ matrix.<key> }}` expression takes each row's value
    under the key, or an empty string for a row declaring no such key, and
    that any other value is taken as written.
    """
    job = Job.model_validate(
        {
            "runs-on"  : "x",
            "strategy" : {
                "matrix": {
                    "include": [
                        {"task": "py:check", "tools": "python uv"},
                        {"tools": "zizmor"}
                    ]
                }
            }
        }
    )

    assert job.expand(value) == values


@mark.parametrize(
    ("paths", "file", "matches"),
    [
        param((), "anything.txt", True, id="no-filter"),
        param((".mise/config.toml",), ".mise/config.toml", True, id="exact"),
        param(
            (".github/actions/provision/**",),
            ".github/actions/provision/action.yml",
            True,
            id = "globstar"
        ),
        param(
            (".github/*",),
            ".github/actions/provision/action.yml",
            False,
            id = "one-segment"
        ),
        param(("pyproject.toml",), "uv.lock", False, id="other-file")
    ]
)
def test_a_trigger_matches_the_files_its_globs_name(
    file    : str,
    matches : bool,
    paths   : tuple[str, ...]
):
    """
    Asserts that a filter matches a file a glob names in full, a `**`
    crossing folders and a `*` staying inside one, and that a trigger naming
    no paths matches every file.
    """
    assert Trigger(paths=paths).matches(file) is matches


@mark.parametrize(
    ("events", "pushes"),
    [
        param("  push:\n", True, id="any-push"),
        param("  push:\n    paths: [uv.lock]\n", True, id="filtered-match"),
        param("  push:\n    paths: [docs/**]\n", False, id="filtered-miss"),
        param("  pull_request:\n", False, id="no-push")
    ]
)
def test_a_workflow_runs_on_a_push_its_filter_matches(events: str, pushes: bool):
    """
    Asserts that a push changing a file runs a workflow only where it runs
    on a push whose filter matches the file or names no paths.
    """
    assert workflow(f"name: w\njobs: {{}}\non:\n{events}").pushes("uv.lock") is pushes


def test_a_label_description_runs_to_100_characters():
    """
    Asserts that a description of 100 characters, the most GitHub's label
    API accepts, validates, whereas one of 101 fails at the description.
    """
    assert len(
        Label.model_validate(ROW | {"description": "x" * 100}).description
    ) == 100

    with raises(ValidationError) as error:
        Label.model_validate(ROW | {"description": "x" * 101})

    assert [(detail["loc"], detail["msg"]) for detail in error.value.errors()] == [
        (("description",), "String should have at most 100 characters")
    ]


@mark.parametrize(
    ("uses", "pin"),
    [
        param(
            "actions/checkout@3d3c42e5",
            ("actions/checkout", "3d3c42e5"),
            id = "pinned"
        ),
        param("$/.github/actions/provision", None, id="self-repository"),
        param("", None, id="shell")
    ]
)
def test_a_step_splits_the_action_from_its_commit(uses: str, pin: tuple | None):
    """
    Asserts that a step naming an action pinned with `@` splits into the
    action and the commit, and that one naming an action of the repository
    or running shell lines holds no pin.
    """
    assert Step(uses=uses).pin == pin


@mark.parametrize(
    ("text", "dependencies"),
    [
        param(
            '#!/bin/sh\n# /// script\n# dependencies = ["minijinja==2.24.0"]\n#\n'
            '# requires-python = ">=3.14"\n# ///\n',
            [Requirement("minijinja==2.24.0")],
            id = "declared"
        ),
        param("#!/bin/sh\n# /// script\n# ///\n", [], id="empty-block"),
        param("#!/bin/sh\ntrue\n", [], id="no-block")
    ]
)
def test_a_task_script_reads_the_requirements_its_metadata_declares(
    dependencies : list[Requirement],
    text         : str,
    tmp_path     : Path
):
    """
    Asserts that a task file's inline metadata reads as the requirements its
    `script` block declares under PEP 723, a comment line holding only `#`
    included, and as none where the file holds no block with a line inside.
    """
    script = tmp_path / "task"
    script.write_text(text)

    assert Task(
        file   = script,
        name   = "t",
        run    = (),
        source = script
    ).dependencies == dependencies


def test_a_label_writes_the_command_that_creates_or_updates_it():
    """
    Asserts that a label writes the `gh label create --force` command
    carrying its color, its description, and its name, which creates the
    label or updates the one already carrying that name.
    """
    assert Label.model_validate(ROW).command == [
        "gh", "label", "create", "--color", "8c055e",
        "--description", "Wrong output", "--force", "🐞 bug"
    ]


@mark.parametrize(
    ("save", "saves"),
    [
        param("'true'", True, id="quoted"),
        param("true", True, id="unquoted"),
        param("'false'", False, id="false")
    ]
)
def test_a_step_reads_its_inputs_as_github_hands_them_to_the_action(
    save  : str,
    saves : bool
):
    """
    Asserts that a step passing `save` as `true`, quoted or not, saves the
    archive, since GitHub hands the action the string either way, and that a
    date passed as an input reads as its text.
    """
    step = workflow(
        text = f"{WORKFLOW}    steps:\n      - uses: $/a\n"
        f"        with: {{ save: {save}, since: 2026-10-01 }}\n"
    ).jobs["a"].steps[0]

    assert (step.saves, step.inputs["since"]) == (saves, "2026-10-01")


def test_a_markdown_file_opening_on_no_front_matter_fails_to_read():
    """
    Asserts that a Markdown file whose text opens on anything but front
    matter fails to read rather than reading as an empty document.
    """
    with raises(ValueError, match="opens on no front matter"):
        Document.load(".md", "name: Bug\n---\n")


def test_a_step_setting_a_number_in_its_env_reads_it_as_a_number():
    """
    Asserts that a step's `env` holding a number, which YAML reads as an
    integer, validates as that integer.
    """
    step = workflow(
        text = f"{WORKFLOW}    steps:\n      - env: {{ RETRIES: 3 }}\n"
    ).jobs["a"].steps[0]

    assert step.env == {"RETRIES": 3}


def test_a_toml_task_declares_no_requirements(tmp_path: Path):
    """
    Asserts that a task declared in TOML, which holds no file of its own,
    declares no requirements.
    """
    assert Task(
        file   = None,
        name   = "t",
        run    = (),
        source = tmp_path
    ).dependencies == []


@mark.parametrize(
    ("details", "message"),
    [
        param(
            "Tasks: a:one, a:two",
            "`mise tasks validate` reports `Alias 'x' is used by multiple tasks` on "
            "`a:one` (`Tasks: a:one, a:two`)",
            id = "details"
        ),
        param(
            "",
            "`mise tasks validate` reports `Alias 'x' is used by multiple tasks` on "
            "`a:one`",
            id = "none"
        )
    ]
)
def test_an_issue_reads_as_a_finding_at_its_severity(details: str, message: str):
    """
    Asserts that a problem mise reports on a task reads as a finding at
    the severity mise gives it, naming the task and the message beside any
    details mise adds.
    """
    issue = Issue(
        details  = details,
        message  = "Alias 'x' is used by multiple tasks",
        severity = Level.WARNING,
        task     = "a:one"
    )

    assert issue.finding(Path("t")) == Finding(
        file    = Path("t"),
        level   = Level.WARNING,
        message = message
    )


@mark.parametrize(
    ("data", "errors"),
    [
        param(
            {"labels": [ROW | {"color": "8C055E"}]},
            [(("labels", 0, "color"), "String should match pattern '^[0-9a-f]{6}$'")],
            id = "uppercase-color"
        ),
        param(
            {"labels": [ROW | {"color": "#8c055e"}]},
            [(("labels", 0, "color"), "String should match pattern '^[0-9a-f]{6}$'")],
            id = "hashed-color"
        ),
        param(
            {"labels": [ROW | {"color": "fff"}]},
            [(("labels", 0, "color"), "String should match pattern '^[0-9a-f]{6}$'")],
            id = "short-color"
        ),
        param(
            {"labels": [ROW | {"glyph": "🐞"}]},
            [(("labels", 0, "glyph"), "Extra inputs are not permitted")],
            id = "further-key"
        ),
        param(
            {"labels": [{"color": "8c055e", "name": "🐞 bug"}]},
            [(("labels", 0, "description"), "Field required")],
            id = "missing-description"
        ),
        param(
            {"categories": [], "labels": [ROW]},
            [(("categories",), "Extra inputs are not permitted")],
            id = "key-beside-the-rows"
        ),
        param(
            {"labels": [ROW, ROW | {"color": "0044aa"}]},
            [(("labels",), "Value error, declares the name `🐞 bug` more than once")],
            id = "repeated-name"
        ),
        param(
            {"labels": [ROW, ROW | {"name": "🧰 tooling"}]},
            [(("labels",), "Value error, declares the color `8c055e` more than once")],
            id = "repeated-color"
        ),
        param(
            {"labels": [ROW, ROW]},
            [
                (
                    ("labels",),
                    "Value error, declares the name `🐞 bug` and the color `8c055e` "
                    "more than once"
                )
            ],
            id = "repeated-row"
        )
    ]
)
def test_a_malformed_registry_fails_at_the_key_it_breaks(
    data   : dict[str, list[dict[str, str]]],
    errors : list[tuple[tuple[int | str, ...], str]]
):
    """
    Asserts that a registry holding a color other than six lowercase hex
    digits, a row with a further key or a missing one, a key beside its
    rows, or two labels sharing a name or a color fails validation at the
    key that breaks the rule, with the message a finding then carries.
    """
    with raises(ValidationError) as error:
        Labels.model_validate(data)

    assert [(detail["loc"], detail["msg"]) for detail in error.value.errors()] == errors


def test_a_warning_prints_as_a_warning_annotation():
    """
    Asserts that a finding at the warning level prints as the `warning`
    workflow command, which annotates the file without failing the run.
    """
    finding = Finding(file=Path("a.toml"), level=Level.WARNING, message="m")

    assert finding.annotation == "::warning file=a.toml::m"


@mark.parametrize(
    ("live", "omitted"),
    [
        param(["🐞 bug"], [], id="all-declared"),
        param(
            ["✨feature", "🐞 bug", "🐞bug"],
            ["✨feature", "🐞bug"],
            id = "older-labels"
        ),
        param([], [], id="none-live")
    ]
)
def test_a_registry_omits_each_live_label_it_declares_nowhere(
    live    : list[str],
    omitted : list[str]
):
    """
    Asserts that a registry names each live label it declares no row for, in
    the order the live labels arrive, and none where it declares them all.
    """
    assert Labels.model_validate({"labels": [ROW]}).omits(live) == omitted


@mark.parametrize(
    ("suffix", "text", "value"),
    [
        param(".toml", 'version = "0.1.0"\n', {"version": "0.1.0"}, id="toml"),
        param(
            ".lock",
            '[[package]]\nname = "a"\n',
            {"package": [{"name": "a"}]},
            id = "lock"
        ),
        param(
            ".yml",
            "on: push\nversion: 0.1.0\n",
            {"on": "push", "version": "0.1.0"},
            id = "yaml"
        ),
        param(".cff", "version: '0.1.0'\n", {"version": "0.1.0"}, id="cff"),
        param(
            ".md",
            "---\nname: Bug\nlabels: []\n---\n\n---\n\nname: Body\n",
            {"name": "Bug", "labels": []},
            id = "front-matter"
        ),
        param(
            ".cff",
            "date-released: 2026-10-02\n",
            {"date-released": "2026-10-02"},
            id = "timestamp"
        )
    ]
)
def test_a_document_reads_the_value_its_suffix_names(
    suffix : str,
    text   : str,
    value  : object
):
    """
    Asserts that a document's text reads as TOML under `.toml` and `.lock`,
    as YAML 1.2 under any other suffix, keeping `on`, a version, and a date
    as the strings they are written as, and as the front matter alone under
    `.md`, ending at its first closing `---`.
    """
    assert Document.load(suffix, text) == value


def test_a_workflow_reads_its_gate_its_rows_and_what_it_saves():
    """
    Asserts that a workflow reads the job under the key `gate`, the jobs it
    waits for, the check run and the task of each of its rows, and whether a
    step passes a composite action `save: 'true'`.
    """
    text = (
        f"{WORKFLOW}    strategy:\n      matrix:\n        include:\n"
        "          - task: lock:check\n    steps:\n"
        "      - uses: $/a\n        with: { save: 'true' }\n"
        "  gate:\n    runs-on: x\n"
    )
    read = workflow(text)

    assert (
        read.gate.image,
        list(read.gated),
        read.checks,
        read.saves("$/a"),
        read.saves("$/b")
    ) == ("x", ["a"], [("", "lock:check")], True, False)
    assert workflow(WORKFLOW).gate is None


@mark.parametrize(
    ("labels", "names"),
    [
        param(["🐞 bug", "🦜 cli"], ["🐞 bug", "🦜 cli"], id="list"),
        param("🐞 bug, 🦜 cli", ["🐞 bug", "🦜 cli"], id="comma-delimited"),
        param("🐞 bug", ["🐞 bug"], id="one"),
        param("", [], id="empty")
    ]
)
def test_a_template_reads_its_labels_from_a_list_or_one_string(
    labels : str | list[str],
    names  : list[str]
):
    """
    Asserts that a template's labels read from a list as written, and from
    one string as each name its commas separate with the spaces around it
    removed, so an empty string names none.
    """
    assert Template.model_validate({"labels": labels}).labels == names


def test_the_lock_maps_each_package_to_its_version():
    """
    Asserts that the lock reads each `[[package]]` row of `uv.lock` as its
    name mapped to the version it resolves.
    """
    rows = {
        "package": [{"name": "a", "version": "1.0"}, {"name": "b", "version": "2.0"}]
    }

    assert Lock.model_validate(rows).versions == {"a": "1.0", "b": "2.0"}


@mark.parametrize(
    ("name", "checks"),
    [
        param(
            "${{ matrix.name }}",
            [("🪵 Lockfiles", "lock:check"), ("🧯 Audit", "repo:audit")],
            id = "matrix"
        ),
        param(
            "Check",
            [("Check", "lock:check"), ("Check", "repo:audit")],
            id = "literal"
        ),
        param("", [("", "lock:check"), ("", "repo:audit")], id="unnamed")
    ]
)
def test_a_job_names_the_check_run_of_each_row_running_a_task(
    checks : list[tuple[str, str]],
    name   : str
):
    """
    Asserts that each row naming a task pairs it with the name its check
    run takes, read through a matrix expression in the job's `name` or as
    written for every row, leaving out a row naming no task.
    """
    job = Job.model_validate(
        {
            "name"     : name,
            "runs-on"  : "x",
            "strategy" : {
                "matrix": {
                    "include": [
                        {"name": "🪵 Lockfiles", "task": "lock:check"},
                        {"name": "🎒 Kit", "tools": "zizmor"},
                        {"name": "🧯 Audit", "task": "repo:audit"}
                    ]
                }
            }
        }
    )

    assert job.checks == checks


@mark.parametrize(
    ("aliases", "command"),
    [
        param(("lockfile",), "mise lockfile", id="alias"),
        param(("lockfile", "lf"), "mise lockfile", id="first-alias"),
        param((), "mise run lock:check", id="none")
    ]
)
def test_a_task_runs_by_its_first_alias_or_by_its_name(
    aliases  : tuple[str, ...],
    command  : str,
    tmp_path : Path
):
    """
    Asserts that a task declaring aliases runs as `mise` and the first of
    them, and that one declaring none runs as `mise run` and its name.
    """
    assert Task(
        aliases = aliases,
        file    = None,
        name    = "lock:check",
        run     = (),
        source  = tmp_path
    ).command == command


def workflow(text: str) -> Workflow:
    """
    Validates `text` as a workflow file's YAML.
    """
    return Workflow.model_validate(text, context={"suffix": ".yml"})
