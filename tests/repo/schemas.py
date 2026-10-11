"""
Pins the records the repository audit and `repo:rulesets` read and report,
meaning the key each document field reads, the path a document's error
carries, what a workflow's jobs, steps, and triggers derive, the annotation
a `Finding` prints at each level, the findings a `Parity` and an `Issue`
make, the requirements a task script declares, the holders a `Manifest`
names, the checks a `Ruleset` requires, the keys `GithubSettings` accepts,
and the command and body each `Request` sends.
"""

from functools import reduce
from json      import loads as from_json
from packaging.requirements import Requirement
from pathlib                import Path
from pydantic               import ValidationError
from pytest                 import Config, mark, param, raises
from tomllib                import loads
from urllib.parse           import unquote

from thermur.repo.schemas import (
    Author,
    Document,
    Finding,
    GithubSettings,
    Issue,
    Job,
    Level,
    Lock,
    Manifest,
    Parity,
    Request,
    Ruleset,
    Step,
    Task,
    Toggles,
    Trigger,
    Workflow
)

WORKFLOW = "name: w\non:\n  push:\njobs:\n  a:\n    runs-on: x\n"


def test_a_bypass_actor_naming_no_mode_sends_no_mode():
    """
    Asserts that a bypass actor naming no `actor_id` or `bypass_mode` is
    sent without either rather than with the record's defaults, so the body
    matches the ruleset as written key for key.
    """
    written = {
        "enforcement"   : "active",
        "name"          : "a",
        "target"        : "branch",
        "bypass_actors" : [{"actor_type": "OrganizationAdmin"}],
        "conditions"    : {"ref_name": {"exclude": [], "include": ["refs/heads/a"]}},
        "rules"         : [{"type": "deletion"}]
    }

    assert Ruleset.model_validate(written).request(None).body == written


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
    Asserts that removing the key a required field's alias names, or the
    key named for the field where it takes no alias, from the checkout's own
    file fails validation at exactly that key, so each field reads the TOML
    key it declares.
    """
    path: list[str] = getattr(
        document.model_fields[field].validation_alias,
        "path",
        [field]
    )
    *table, key = path
    parsed      = loads(
        (pytestconfig.rootpath / document.model_config["title"]).read_text()
    )
    del reduce(dict.__getitem__, table, parsed)[key]

    with raises(ValidationError) as error:
        document.model_validate(parsed)

    assert [detail["loc"] for detail in error.value.errors()] == [tuple(path)]


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

    assert (job.expand(value), job.tasks) == (values, ["py:check"])


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


def test_a_required_check_naming_no_app_sends_no_app():
    """
    Asserts that a required check naming no `integration_id` is sent without
    one rather than as `null`, which the rulesets API reads as no integer,
    so the body matches the ruleset as written key for key.
    """
    written = {
        "enforcement"   : "active",
        "name"          : "a",
        "target"        : "branch",
        "bypass_actors" : [],
        "conditions"    : {"ref_name": {"exclude": [], "include": ["refs/heads/a"]}},
        "rules"         : [
            {
                "type"       : "required_status_checks",
                "parameters" : {
                    "strict_required_status_checks_policy" : True,
                    "required_status_checks"               : [{"context": "c"}]
                }
            }
        ]
    }

    assert Ruleset.model_validate(written).request(None).body == written


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


def test_a_step_setting_a_number_in_its_env_reads_it_as_a_number():
    """
    Asserts that a step's `env` holding a number, which YAML reads as an
    integer, validates as that integer.
    """
    step = workflow(
        text = f"{WORKFLOW}    steps:\n      - env: {{ RETRIES: 3 }}\n"
    ).jobs["a"].steps[0]

    assert step.env == {"RETRIES": 3}


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


def test_a_toggle_turns_its_endpoint_on_or_off():
    """
    Asserts that each toggle reads as the endpoint it names, which `PUT`
    turns on and `DELETE` turns off, in the order the record declares its
    fields whatever order the table lists them in.
    """
    toggles = Toggles.model_validate(
        {
            "automated-security-fixes"        : False,
            "immutable-releases"              : True,
            "private-vulnerability-reporting" : True,
            "vulnerability-alerts"            : False
        }
    )

    assert [(call.method, call.endpoint) for call in toggles.requests] == [
        ("DELETE", "automated-security-fixes"),
        ("PUT", "immutable-releases"),
        ("PUT", "private-vulnerability-reporting"),
        ("DELETE", "vulnerability-alerts")
    ]


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


def test_a_warning_prints_as_a_warning_annotation():
    """
    Asserts that a finding at the warning level prints as the `warning`
    workflow command, which annotates the file without failing the run.
    """
    finding = Finding(file=Path("a.toml"), level=Level.WARNING, message="m")

    assert finding.annotation == "::warning file=a.toml::m"


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
    ("call", "command", "payload"),
    [
        param(
            Request(body={"names": ["jax"]}, endpoint="topics", method="PUT"),
            [
                "gh", "api", "--input", "-",
                "--method", "PUT", "--silent", "repos/{owner}/{repo}/topics"
            ],
            b'{"names":["jax"]}',
            id = "body"
        ),
        param(
            Request(body={}, endpoint="", method="PATCH"),
            [
                "gh", "api", "--input", "-",
                "--method", "PATCH", "--silent", "repos/{owner}/{repo}"
            ],
            b"{}",
            id = "empty-body"
        ),
        param(
            Request(endpoint="vulnerability-alerts", method="DELETE"),
            [
                "gh", "api", "--method",
                "DELETE", "--silent", "repos/{owner}/{repo}/vulnerability-alerts"
            ],
            None,
            id = "no-body"
        )
    ]
)
def test_a_request_sends_its_body_on_standard_input(
    call    : Request,
    command : list[str],
    payload : bytes | None
):
    """
    Asserts that a request carrying a body, an empty one included, reads
    it from standard input as compact JSON, that one carrying none reads
    nothing, and that an empty endpoint reaches the repository itself.
    """
    assert (call.command, call.payload) == (command, payload)


@mark.parametrize(
    ("live", "method", "endpoint"),
    [
        param(24390266, "PUT", "rulesets/24390266", id="update"),
        param(None, "POST", "rulesets", id="create")
    ]
)
def test_a_ruleset_updates_its_live_ruleset_or_creates_one(
    endpoint     : str,
    live         : int | None,
    method       : str,
    pytestconfig : Config
):
    """
    Asserts that a ruleset updates the live ruleset whose id it is handed
    in place and creates itself where it is handed none, sending exactly the
    JSON its file holds, with no default filled in.
    """
    path    = Path(".github/rulesets/main.json")
    request = Ruleset.read(pytestconfig.rootpath, path).request(live)

    assert (request.method, request.endpoint, request.body) == (
        method, endpoint,
        from_json((pytestconfig.rootpath / path).read_text())
    )


@mark.parametrize(
    ("name", "contexts"),
    [param("main", {"📡 Readout"}, id="main"), param("tags", set(), id="tags")]
)
def test_a_ruleset_reads_the_checks_it_requires(
    contexts     : set[str],
    name         : str,
    pytestconfig : Config
):
    """
    Asserts that the `main` ruleset requires the `📡 Readout` check alone
    and the tag ruleset, holding no `required_status_checks` rule, requires
    none.
    """
    path = Path(f".github/rulesets/{name}.json")

    assert Ruleset.read(pytestconfig.rootpath, path).contexts == contexts


@mark.parametrize(
    ("old", "new", "locations"),
    [
        param("[token]", "[tokens]", [("token",), ("tokens",)], id="table"),
        param(
            "has_wiki",
            "has_wiky",
            [("repository", "has_wiki"), ("repository", "has_wiky")],
            id = "key"
        ),
        param(
            "immutable-releases",
            "immutable_releases",
            [("toggles", "immutable-releases"), ("toggles", "immutable_releases")],
            id = "toggle"
        ),
        param(
            "sha_pinning_required",
            "sha_pining_required",
            [("actions", "sha_pinning_required"), ("actions", "sha_pining_required")],
            id = "actions"
        ),
        param(
            "secret_scanning ",
            "secret_scaning ",
            [
                ("repository", "security_and_analysis", "secret_scanning"),
                ("repository", "security_and_analysis", "secret_scaning")
            ],
            id = "analysis"
        ),
        param(
            '{ status = "enabled" }\nsecret',
            '{ state = "enabled" }\nsecret',
            [
                ("repository", "security_and_analysis", "secret_scanning", "status"),
                ("repository", "security_and_analysis", "secret_scanning", "state")
            ],
            id = "feature"
        ),
        param(
            "default_workflow_permissions",
            "default_workflow_permisions",
            [
                ("token", "default_workflow_permissions"),
                ("token", "default_workflow_permisions")
            ],
            id = "token"
        )
    ]
)
def test_a_misspelled_table_or_key_fails_the_settings(
    locations    : list[tuple[str, ...]],
    new          : str,
    old          : str,
    pytestconfig : Config
):
    """
    Asserts that a misspelled table, a misspelled key in each table or in
    a feature it nests, and a toggle spelled with the underscore its field
    takes rather than the endpoint's hyphen each fail validation, naming the
    setting missing beside the one the file carries in its place.
    """
    text = (pytestconfig.rootpath / ".github/settings.toml").read_text()

    assert old in text
    with raises(ValidationError) as error:
        GithubSettings.model_validate(
            text.replace(old, new),
            context = {"suffix": ".toml"}
        )

    assert [detail["loc"] for detail in error.value.errors()] == locations


def test_a_workflow_reads_its_gate_its_rows_and_what_it_saves():
    """
    Asserts that a workflow reads the job under the key `gate`, the jobs it
    waits for, the tasks its rows run, and whether a step passes a composite
    action `save: 'true'`.
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
        read.tasks,
        read.saves("$/a"),
        read.saves("$/b")
    ) == ("x", ["a"], ["lock:check"], True, False)
    assert workflow(WORKFLOW).gate is None


@mark.parametrize(
    ("events", "pulls"),
    [
        param("  pull_request:\n", True, id="pull-request"),
        param("  pull_request:\n  push:\n", True, id="both"),
        param("  push:\n", False, id="push")
    ]
)
def test_a_workflow_reads_whether_it_runs_on_a_pull_request(events: str, pulls: bool):
    """
    Asserts that a workflow runs on a pull request only where `pull_request`
    is among its events, whatever else it runs on.
    """
    assert workflow(f"name: w\njobs: {{}}\non:\n{events}").on_pull_request is pulls


def test_the_lock_maps_each_package_to_its_version():
    """
    Asserts that the lock reads each `[[package]]` row of `uv.lock` as its
    name mapped to the version it resolves.
    """
    rows = {
        "package": [{"name": "a", "version": "1.0"}, {"name": "b", "version": "2.0"}]
    }

    assert Lock.model_validate(rows).versions == {"a": "1.0", "b": "2.0"}


def workflow(text: str) -> Workflow:
    """
    Validates `text` as a workflow file's YAML.
    """
    return Workflow.model_validate(text, context={"suffix": ".yml"})
