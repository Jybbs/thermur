"""
Pins what each check `repo:audit` runs reports over a copy of the checkout,
nothing for the copy as it stands and one finding for each way a case breaks
it.
"""

from common.edits import append, edit, task
from pathlib      import Path
from pydantic     import TypeAdapter, ValidationError
from pytest       import MonkeyPatch, mark, param, raises
from shutil       import copystat

from thermur.repo.checkout import Checkout
from thermur.repo.checks   import (
    AnchorCheck,
    BinCheck,
    Check,
    CommitCheck,
    GateCheck,
    ImageCheck,
    KitCheck,
    ParityCheck,
    PinCheck,
    ProvisionCheck,
    RulesetCheck,
    RunCheck,
    ScriptCheck,
    SummaryCheck,
    TaskCheck,
    WheelCheck
)
from thermur.repo.schemas  import Finding, Issue, Level, Task

ACTION = ".github/actions/provision/action.yml"
BRIEF  = ".mise/tasks/gha/brief.py"
CI     = ".github/workflows/ci.yml"
DRAFT  = ".github/workflows/draft.yml"
FLOOR  = "the floor of `requires-python` in `pyproject.toml` reads `3.14`"
MAIN   = ".github/rulesets/main.json"
SHELL  = "#!/usr/bin/env -S bash -euo pipefail\n"
WARM   = ".github/workflows/warm.yml"


def test_a_check_declaring_nothing_makes_no_finding(checkout: Checkout):
    """
    Asserts that the base check, comparing no restated declarations and
    scanning nothing of its own, makes no finding, the default each check
    holding pairs or a scan of its own builds on.
    """
    assert Check(checkout=checkout).findings == []


def test_a_copy_differing_byte_for_byte_is_named(checkout: Checkout):
    """
    Asserts that a program whose bytes differ from the first program's is
    named against that first program, even where the two match in size and
    in modification time.
    """
    edit(checkout, ".mise/bin/pytest", new="--frozen", old="--locked")
    copystat(checkout.root / ".mise/bin/prose", checkout.root / ".mise/bin/pytest")

    assert BinCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(".mise/bin/pytest"),
            message = "`.mise/bin/pytest` differs byte for byte from `.mise/bin/prose`"
        )
    ]


@mark.parametrize(
    ("path", "old", "new", "message"),
    [
        param(
            "README.md",
            "python-3.14+",
            "python-3.13+",
            f"The Python badge in `README.md` reads `3.13`, where {FLOOR}",
            id = "python-badge"
        ),
        param(
            "README.md",
            "badge/python-3.14+",
            "badge/py-3.14+",
            f"The Python badge in `README.md` is missing, where {FLOOR}",
            id = "python-badge-missing"
        ),
        param(
            "pyproject.toml",
            'target-version = "3.14"',
            'target-version = "3.13"',
            f"The `target-version` in `[tool.prose]` reads `3.13`, where {FLOOR}",
            id = "target-version"
        ),
        param(
            ".mise/config.toml",
            'python = "3.14.8"',
            'python = "3.15.0"',
            "The minor of the `python` pin in `.mise/config.toml` reads `3.15`, "
            f"where {FLOOR}",
            id = "mise-python"
        ),
        param(
            "pyproject.toml",
            '"==0.12.22"',
            '"==0.12.21"',
            "The `required-version` in `[tool.uv]` reads `==0.12.21`, where the "
            "`uv` pin in `.mise/config.toml` as a `==` requirement reads "
            "`==0.12.22`",
            id = "required-version"
        ),
        param(
            "README.md",
            "License-MIT-",
            "License-Apache--2.0-",
            "The License badge in `README.md` reads `Apache-2.0`, where the "
            "`license` in `pyproject.toml` reads `MIT`",
            id = "license-badge"
        ),
        param(
            "LICENSE",
            "MIT License",
            "BSD License",
            "The title of `LICENSE` reads `BSD License`, where the full name of the "
            "`license` in `pyproject.toml` reads `MIT License`",
            id = "license-title"
        ),
        param(
            "LICENSE",
            "James Parkington",
            "Jybbs",
            "The copyright holder list in `LICENSE` reads `Jybbs`, where the author "
            "list in `pyproject.toml` reads `James Parkington`",
            id = "license-holders"
        )
    ]
)
def test_a_restated_declaration_that_drifts_is_named(
    checkout : Checkout,
    path     : str,
    old      : str,
    new      : str,
    message  : str
):
    """
    Asserts that each file restating the Python version, the uv release,
    or the license is named once its copy drifts from the declaration it
    restates, or once the file stops carrying it.
    """
    edit(checkout, path, new=new, old=old)

    assert ParityCheck(checkout=checkout).findings == [
        Finding(file=Path(path), message=message)
    ]


def test_a_copyright_year_range_still_reads_its_holders(checkout: Checkout):
    """
    Asserts that a copyright line naming a range of years, as a license
    carried past its first year does, reads the holders after the range.
    """
    edit(checkout, "LICENSE", new="Copyright (c) 2025-2026", old="Copyright (c) 2025")

    assert ParityCheck(checkout=checkout).findings == []


@mark.parametrize(
    ("text", "named"),
    [
        param(f"{SHELL}uv run pytest\n", True, id="bare"),
        param("#!/usr/bin/env -S uv run\nprint()\n", True, id="shebang"),
        param(f"{SHELL}uv run --exact pytest\n", True, id="exact-alone"),
        param(f"{SHELL}uv run --locked --exact pytest\n", True, id="reordered"),
        param(f"{SHELL}uv run --exact --lockedx pytest\n", True, id="partial-flag"),
        param(f'{SHELL}report=$(uv run pytest "$@")\n', True, id="substitution"),
        param(f"{SHELL}uv run --exact --locked pytest -x\n", False, id="locked"),
        param(
            f"{SHELL}uv run \\\n  --exact \\\n  --locked \\\n  pytest\n",
            False,
            id = "continued"
        ),
        param(
            f'{SHELL}#MISE description = "Wrap `uv run`"\nuv lock --check\n',
            False,
            id = "comment"
        ),
        param(
            f"{SHELL}if true; then\n  # uv run pytest\n  :\nfi\n",
            False,
            id = "indented-comment"
        ),
        param(
            '#!/bin/sh\nexec uv run --exact --locked --project "$root" "$@"\n',
            False,
            id = "program-script"
        )
    ]
)
def test_a_task_is_named_where_its_uv_run_is_not_locked(
    checkout : Checkout,
    text     : str,
    named    : bool
):
    """
    Asserts that a task whose `uv run` does not open on `--exact --locked`,
    in that order and as whole flags, is named, whereas one opening every
    `uv run` on them, across backslash continuations or beside further
    flags, passes, as does one naming `uv run` only in a comment.
    """
    path = task(checkout, "py/sweep", text)

    assert RunCheck(checkout=checkout).findings == [
        Finding(
            file    = path,
            message = f"`{path}` runs `uv run` without `--exact --locked` opening its "
            "arguments"
        )
    ] * named


def test_a_gate_running_under_a_bare_always_passes(checkout: Checkout):
    """
    Asserts that a gate whose condition is `always()` written without the
    `${{ }}` GitHub lets a condition leave out passes.
    """
    edit(checkout, CI, new="if      : always()", old="if      : ${{ always() }}")

    assert GateCheck(checkout=checkout).findings == []


def test_a_hatchling_pin_in_another_spelling_passes(checkout: Checkout):
    """
    Asserts that an exact pin naming hatchling in another spelling its
    normalized name allows passes, since a package's name compares case
    insensitively.
    """
    edit(
        checkout,
        "pyproject.toml",
        new = '"Hatchling==1.32.4"',
        old = '"hatchling==1.32.4"'
    )

    assert PinCheck(checkout=checkout).findings == []


def test_a_job_on_a_second_runner_image_is_named(checkout: Checkout):
    """
    Asserts that each job running on an image other than the first job's is
    named against that first job, in path order.
    """
    edit(checkout, WARM, new="&image ubuntu-24.04", old="&image ubuntu-26.04")

    assert [finding.message for finding in ImageCheck(checkout=checkout).findings] == [
        f"The runner image of the `{key}` job in `{WARM}` reads `ubuntu-24.04`, where "
        f"the one the `check` job in `{CI}` runs on reads `ubuntu-26.04`"
        for key in checkout.workflows[Path(WARM)].jobs
    ]


@mark.parametrize(
    "check",
    [param(check, id=check.__name__) for check in Check.__subclasses__()]
)
def test_each_check_passes_the_checkout_as_it_stands(
    check    : type[Check],
    checkout : Checkout
):
    """
    Asserts that every check makes no finding over an unedited copy of the
    checkout.
    """
    assert check(checkout=checkout).findings == []


@mark.parametrize(
    ("target", "named"),
    [
        param("thermur.missing:app", True, id="missing"),
        param("thermur.repo.checks:Check", False, id="module"),
        param("thermur.cli:app", False, id="package")
    ]
)
def test_a_script_is_named_where_its_module_is_not_carried(
    checkout : Checkout,
    target   : str,
    named    : bool
):
    """
    Asserts that a `[project.scripts]` target is named where the wheel
    carries no module for it, and passes where it names a module or a
    package, the package resolving to its `__init__.py`.
    """
    (checkout.root / "src/thermur/cli").mkdir()
    (checkout.root / "src/thermur/cli/__init__.py").write_text("app = None\n")
    append(checkout, "pyproject.toml", f'\n[project.scripts]\nthermur = "{target}"\n')

    assert WheelCheck(checkout=checkout).findings == [
        Finding(
            file    = Path("pyproject.toml"),
            message = f"The `thermur` script targets `{target.partition(':')[0]}`, "
            "which the wheel does not carry"
        )
    ] * named


def test_a_linked_program_is_named(checkout: Checkout):
    """
    Asserts that a program written as a symlink is named once, and is left
    out of the byte comparison it would otherwise fail.
    """
    linked = checkout.root / ".mise/bin/pytest"
    linked.unlink()
    linked.symlink_to("../config.toml")

    assert BinCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(".mise/bin/pytest"),
            message = "`.mise/bin/pytest` is a symlink rather than a copy"
        )
    ]


@mark.parametrize(
    ("old", "new", "message"),
    [
        param(
            '"hatchling==1.32.4"',
            '"hatchling>=1.32.4"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-range"
        ),
        param(
            '"hatchling==1.32.4"',
            '"hatchling==1.32.*"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-wildcard"
        ),
        param(
            '"hatchling==1.32.4"',
            '"setuptools==80.9.0"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-absent"
        ),
        param(
            '"pluggy==1.6.0"',
            '"pluggy>=1.6,<2"',
            "The build constraint `pluggy<2,>=1.6` in `[tool.uv]` is not an exact pin",
            id = "constraint-range"
        ),
        param(
            '"hatchling==1.32.4"',
            '"hatchling"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-unpinned"
        ),
        param(
            '"pluggy==1.6.0"',
            '"pluggy"',
            "The build constraint `pluggy` in `[tool.uv]` is not an exact pin",
            id = "constraint-unpinned"
        ),
        param(
            '"hatchling==1.32.4"',
            '"hatchling==1.32.4,<2"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-second-specifier"
        )
    ]
)
def test_a_build_requirement_naming_a_range_is_named(
    checkout : Checkout,
    old      : str,
    new      : str,
    message  : str
):
    """
    Asserts that a hatchling requirement naming a range, a wildcard, or no
    hatchling at all, and a build constraint naming a range, are each named,
    since uv resolves both outside `uv.lock`.
    """
    edit(checkout, "pyproject.toml", new=new, old=old)

    assert PinCheck(checkout=checkout).findings == [
        Finding(file=Path("pyproject.toml"), message=message)
    ]


@mark.parametrize(
    ("path", "old", "new", "messages"),
    [
        param(
            "pyproject.toml",
            'requires-python = ">=3.14"\n',
            "",
            ["`project.requires-python` in `pyproject.toml`: Field required"],
            id = "missing-field"
        ),
        param(
            "pyproject.toml",
            'license         = "MIT"\nname            = "thermur"\n'
            'readme          = "README.md"\nrequires-python = ">=3.14"\n',
            'name            = "thermur"\n',
            [
                "`project.requires-python` in `pyproject.toml`: Field required",
                "`project.license` in `pyproject.toml`: Field required",
                "`project.readme` in `pyproject.toml`: Field required"
            ],
            id = "missing-fields"
        ),
        param(
            "pyproject.toml",
            '"hatchling==1.32.4"',
            '"hatchling=1.32.4"',
            [
                "`build-system.requires.0` in `pyproject.toml`: Value error, Expected "
                "semicolon (after name with no version specifier) or end\n    "
                "hatchling=1.32.4\n             ^"
            ],
            id = "malformed-requirement"
        ),
        param(
            ".mise/config.toml",
            'python = "3.14.8"',
            'python = "3"',
            [
                "`tools.python` in `.mise/config.toml`: Value error, names no major "
                "and minor version"
            ],
            id = "unversioned-python"
        )
    ]
)
def test_a_document_failing_validation_is_reported_rather_than_raised(
    checkout : Checkout,
    path     : str,
    old      : str,
    new      : str,
    messages : list[str]
):
    """
    Asserts that a document missing fields the checks read, or holding one
    that fails to parse, makes one finding on that document per field rather
    than ending the audit, whichever field the check itself reads.
    """
    edit(checkout, path, new=new, old=old)

    assert ParityCheck(checkout=checkout).findings == [
        Finding(file=Path(path), message=message) for message in messages
    ]


@mark.parametrize(
    ("character", "problem"),
    [
        param(
            "\x85",
            "while scanning a simple key, could not find expected ':'",
            id = "next-line"
        ),
        param(
            "\x07",
            "unacceptable character #x0007: special characters are not allowed",
            id = "control-character"
        )
    ]
)
def test_a_manifest_pyyaml_cannot_parse_is_named_for_its_problem(
    character : str,
    checkout  : Checkout,
    problem   : str
):
    """
    Asserts that a manifest yaml-rs reads but PyYAML cannot parse, here one
    whose description holds the NEL that YAML 1.1 reads as a line break or
    a control character PyYAML's reader rejects, is named for the problem
    PyYAML reports rather than raising.
    """
    edit(
        checkout,
        ACTION,
        new = f"The number{character}of commits",
        old = "The number of commits"
    )

    assert AnchorCheck(checkout=checkout).findings == [
        Finding(file=Path(ACTION), message=f"PyYAML cannot parse `{ACTION}`: {problem}")
    ]


@mark.parametrize(
    "run",
    [
        param('"uv run pytest"', id="line"),
        param('["uv run pytest", { task = "py:check" }]', id="beside-a-reference")
    ]
)
def test_a_toml_task_running_uv_run_unlocked_is_named_on_its_configuration(
    checkout : Checkout,
    run      : str
):
    """
    Asserts that a task `.mise/config.toml` declares in TOML is held to the
    lines it runs and named on that configuration, including a line beside
    a table naming another task, which is held to that line rather than
    failing validation.
    """
    append(checkout, ".mise/config.toml", f'\n[tasks."py:sweep"]\nrun = {run}\n')

    assert [finding.file for finding in RunCheck(checkout=checkout).findings] == [
        Path(".mise/config.toml")
    ]


def test_a_moved_python_floor_names_every_copy(checkout: Checkout):
    """
    Asserts that raising `requires-python` names the README's badge, the
    formatter's `target-version`, and the minor mise pins, each against the
    new floor.
    """
    edit(checkout, "pyproject.toml", new='">=3.15"', old='">=3.14"')

    assert [finding.file for finding in ParityCheck(checkout=checkout).findings] == [
        Path("README.md"),
        Path("pyproject.toml"),
        Path(".mise/config.toml")
    ]


def test_a_packages_entry_holding_no_module_is_named(checkout: Checkout):
    """
    Asserts that a wheel `packages` entry naming a folder that holds no
    module, which hatchling skips while still writing the wheel, is named.
    """
    edit(
        checkout = checkout,
        new      = 'packages = ["src/config", "src/thermur"]',
        old      = 'packages = ["src/thermur"]',
        path     = "pyproject.toml"
    )

    assert WheelCheck(checkout=checkout).findings == [
        Finding(
            file    = Path("pyproject.toml"),
            message = "The wheel `packages` entry `src/config` holds no module"
        )
    ]


def test_a_problem_on_a_task_outside_the_checkout_is_left_out(
    checkout    : Checkout,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a problem `mise tasks validate` reports on a task no file
    of the checkout declares, such as one a global configuration adds, makes
    no finding.
    """
    outside = Issue(message="m", severity=Level.ERROR, task="global:task")
    monkeypatch.setattr(Checkout, "issues", property(lambda checkout: [outside]))

    assert TaskCheck(checkout=checkout).findings == []


def test_a_program_on_the_path_running_uv_run_unlocked_is_named(checkout: Checkout):
    """
    Asserts that a program under `.mise/bin/` is held to `--exact --locked`
    as a task is.
    """
    edit(checkout, ".mise/bin/pytest", new="", old="--exact --locked ")

    assert [finding.file for finding in RunCheck(checkout=checkout).findings] == [
        Path(".mise/bin/pytest")
    ]


def test_a_python_task_is_read_by_its_shebang_alone(checkout: Checkout):
    """
    Asserts that a Python task is held to the `uv run` its shebang runs and
    not to one its docstring mentions.
    """
    docstring = '"""\nWraps `uv run pytest`.\n"""\n'
    task(
        checkout = checkout,
        name     = "repo/sweep.py",
        text     = f"#!/usr/bin/env -S uv run --exact --locked\n{docstring}"
    )
    path = task(checkout, "repo/tally.py", "#!/usr/bin/env -S uv run\n")

    assert [finding.file for finding in RunCheck(checkout=checkout).findings] == [path]


@mark.parametrize(
    ("old", "new", "message"),
    [
        param("  gate:\n", "  brief:\n", f"`{CI}` ends on no `gate` job", id="no-gate"),
        param(
            "    name    : 📡 Readout\n",
            "",
            f"The `gate` job in `{CI}` sets no name",
            id = "unnamed"
        ),
        param(
            "${{ always() }}",
            "${{ success() }}",
            f"The `gate` job in `{CI}` runs under `${{{{ success() }}}}` rather than "
            "`always()`",
            id = "condition"
        ),
        param(
            "needs   : [check]",
            "needs   : []",
            f"The `gate` job in `{CI}` does not wait for `check`",
            id = "unwaited"
        )
    ]
)
def test_a_gate_that_cannot_carry_every_verdict_is_named(
    checkout : Checkout,
    message  : str,
    new      : str,
    old      : str
):
    """
    Asserts that a workflow ending on no `gate` job is named, as is a gate
    job that sets no name, runs under a condition other than `always()`, or
    leaves a job unwaited.
    """
    edit(checkout, CI, new=new, old=old)

    assert GateCheck(checkout=checkout).findings == [
        Finding(file=Path(CI), message=message)
    ]


@mark.parametrize(
    ("path", "old", "new", "context"),
    [
        param(
            MAIN,
            '"context": "📡 Readout"',
            '"context": "🐦‍⬛ CI"',
            "🐦‍⬛ CI",
            id = "required"
        ),
        param(
            CI,
            "name    : 📡 Readout",
            "name    : 📡 Report",
            "📡 Readout",
            id = "gate"
        ),
        param(CI, "  gate:\n", "  brief:\n", "📡 Readout", id="no-gate")
    ]
)
def test_a_required_check_no_pull_request_gate_names_is_named(
    checkout : Checkout,
    context  : str,
    new      : str,
    old      : str,
    path     : str
):
    """
    Asserts that renaming the check the `main` ruleset requires, renaming
    the gate of the one workflow running on a pull request, or ending that
    workflow on no gate names the required check on the ruleset, although
    the gates of the push workflows still carry the name `📡 Readout`.
    """
    edit(checkout, path, new=new, old=old)

    assert RulesetCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(MAIN),
            message = f"`{MAIN}` requires the `{context}` check, which no gate of a "
            "workflow running on a pull request names"
        )
    ]


def test_a_ruleset_failing_validation_is_reported_on_its_path(checkout: Checkout):
    """
    Asserts that a ruleset carrying a key the rulesets API does not take
    is named on its own file, one finding per field, rather than ending
    the audit.
    """
    tags = ".github/rulesets/tags.json"
    edit(checkout, tags, new='"enforcment"', old='"enforcement"')

    assert RulesetCheck(checkout=checkout).findings == [
        Finding(file=Path(tags), message=f"`enforcement` in `{tags}`: Field required"),
        Finding(
            file    = Path(tags),
            message = f"`enforcment` in `{tags}`: Extra inputs are not permitted"
        )
    ]


def test_a_task_failing_validation_is_named_at_mise_s_severity(checkout: Checkout):
    """
    Asserts that a problem `mise tasks validate` reports on a task, here
    an alias two tasks share, is named on the file holding the task at the
    severity mise reports.
    """
    path = task(checkout, "py/sweep", f'{SHELL}#MISE alias = "audit"\ntrue\n')

    assert [
        (finding.file, finding.level)
        for finding in TaskCheck(checkout=checkout).findings
    ] == [(path, Level.ERROR)]


@mark.parametrize(
    ("old", "new", "finding"),
    [
        param(
            "          - name  : 🎒 Kit · Workflows\n            tools : zizmor\n",
            "",
            Finding(
                file    = Path(CI),
                message = f"`{CI}` installs the `zizmor` subset, whose archive no job "
                "saves"
            ),
            id = "unsaved"
        ),
        param(
            "            tools : zizmor\n",
            "            tools : zizmor\n"
            "          - name  : 🎒 Kit · Site\n"
            "            tools : bun\n",
            Finding(
                file    = Path(WARM),
                message = f"`{WARM}` saves the archive of the `bun` subset, which no "
                "job installs"
            ),
            id = "unread"
        ),
        param(
            "            tools : zizmor\n",
            "            tool  : zizmor\n",
            Finding(
                file    = Path(CI),
                message = f"`{CI}` installs the `zizmor` subset, whose archive no job "
                "saves"
            ),
            id = "undeclared"
        )
    ]
)
def test_a_subset_installed_or_saved_alone_is_named(
    checkout : Checkout,
    finding  : Finding,
    new      : str,
    old      : str
):
    """
    Asserts that a subset the jobs install that no job saves, and a subset
    a job saves that no job installs, are each named on the workflow naming
    it, a Kit row declaring no `tools` saving no subset.
    """
    edit(checkout, WARM, new=new, old=old)

    assert KitCheck(checkout=checkout).findings == [finding]


def test_a_tools_input_to_another_action_installs_no_subset(checkout: Checkout):
    """
    Asserts that a `tools` input a step passes an action other than a
    composite action of the checkout names no subset a Kit job must save.
    """
    edit(
        checkout = checkout,
        new      = "uses : actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n"
        "        with : { tools: bun }",
        old  = "uses : $/.github/actions/provision\n        with : { depth: '2' }",
        path = DRAFT
    )

    assert KitCheck(checkout=checkout).findings == []


@mark.parametrize(
    ("old", "new", "left"),
    [
        param(", .mise/mise.lock]", "]", [".mise/mise.lock"], id="filtered-out"),
        param(
            ".mise/config.toml, .mise/mise.lock]",
            ".mise/**]",
            [],
            id = "globbed"
        ),
        param(
            "  push:\n    branches",
            "  pull_request:\n    branches",
            [".mise/config.toml", ".mise/mise.lock"],
            id = "no-push"
        )
    ]
)
def test_a_hashed_file_no_push_warms_is_named(
    checkout : Checkout,
    left     : list[str],
    new      : str,
    old      : str
):
    """
    Asserts that each file the composite action hashes into an archive's key
    is named where a push changing it runs no workflow saving that archive,
    and that a glob covering both files passes.
    """
    edit(checkout, WARM, new=new, old=old)

    assert ProvisionCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(WARM),
            message = f"The push filter of `{WARM}` leaves out `{file}`, which "
            f"`{ACTION}` hashes"
        )
        for file in left
    ]


@mark.parametrize(
    ("new", "message"),
    [
        param(
            "minijinja==2.23.0",
            f"The pin on `minijinja` in `{BRIEF}` reads `==2.23.0`, where the version "
            "`uv.lock` resolves reads `==2.24.0`",
            id = "drifted"
        ),
        param(
            "minijinja",
            f"The pin on `minijinja` in `{BRIEF}` is missing, where the version "
            "`uv.lock` resolves reads `==2.24.0`",
            id = "unpinned"
        ),
        param("rich==14.0.0", None, id="unlocked")
    ]
)
def test_a_script_pinning_another_version_than_the_lock_is_named(
    checkout : Checkout,
    message  : str | None,
    new      : str
):
    """
    Asserts that a task script pinning another version of a package
    `uv.lock` resolves, or pinning none, is named, and that one requiring a
    package `uv.lock` does not resolve passes.
    """
    edit(checkout, BRIEF, new=new, old="minijinja==2.24.0")

    assert [finding.message for finding in ScriptCheck(checkout=checkout).findings] == (
        [message] if message else []
    )


@mark.parametrize(
    ("path", "old", "new", "subject"),
    [
        param(
            CI,
            'run  : mise run "$TASK"',
            'run  : mise run "$TASK" >> "$GITHUB_STEP_SUMMARY"',
            f"The `check` job in `{CI}`",
            id = "variable"
        ),
        param(
            DRAFT,
            "run  : ./.mise/tasks/gha/cut.py",
            "run  : mise run gha:brief",
            f"The `cut` job in `{DRAFT}`",
            id = "brief"
        ),
        param(
            ACTION,
            '>> "$GITHUB_OUTPUT"',
            '>> "$GITHUB_STEP_SUMMARY"',
            f"The composite action `{ACTION}`",
            id = "action"
        )
    ]
)
def test_a_step_outside_a_gate_writing_the_summary_is_named(
    checkout : Checkout,
    new      : str,
    old      : str,
    path     : str,
    subject  : str
):
    """
    Asserts that a job other than a gate, and a composite action, are each
    named where a step writes to the workflow step summary, by the variable
    holding its path or by running `gha:brief`.
    """
    edit(checkout, path, new=new, old=old)

    assert SummaryCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(path),
            message = f"{subject} writes to the workflow step summary, which only a "
            "`gate` job writes"
        )
    ]


@mark.parametrize(
    ("prefix", "names"),
    [
        param("probe: &x 1\n", ["x"], id="value"),
        param("probe: &x 1\nalso: *x\nagain: *x\n", ["x"], id="aliased-twice"),
        param("--- &doc\n", ["doc"], id="document"),
        param("probe:\n  - &k run: |\n      echo\n", ["k"], id="key-opening-a-block"),
        param("probe: !!str &x 1\n", ["x"], id="tagged"),
        param("probe: {k: &x v, j: *x}\n", ["x"], id="flow"),
        param("probe: x, 'y\nalso: &x 1\nagain: 'z'\n", ["x"], id="quote-inside-prose"),
        param("probe: &x 1\nalso: &y 2\n", ["x", "y"], id="in-order"),
        param('probe: "x: &y"\n', [], id="double-quoted"),
        param("probe: 'p: &q'\n", [], id="single-quoted"),
        param("probe: 1  # c: &r\n", [], id="comment"),
        param("probe: this & that\n", [], id="prose"),
        param("probe: |\n  echo &x\n  - key: &y\n", [], id="block-scalar"),
        param("probe: ${{ a && b }}\n", [], id="expression")
    ]
)
def test_each_anchor_a_manifest_carries_is_named_once(
    checkout : Checkout,
    names    : list[str],
    prefix   : str
):
    """
    Asserts that a composite action's manifest is named once for each anchor
    it declares or reads through an alias, in the order they appear, and
    that an `&` inside a quoted scalar, a comment, prose, a block scalar, or
    a `${{ }}` expression passes.
    """
    edit(checkout, ACTION, new=f"{prefix}name        :", old="name        :")

    assert AnchorCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(ACTION),
            message = f"`{ACTION}` carries the YAML anchor `&{name}`, which GitHub's "
            "action-manifest parser rejects"
        )
        for name in names
    ]


def test_a_validation_error_outside_the_documents_is_raised(
    checkout    : Checkout,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a task listing failing validation, whose error names no
    file of the checkout, ends the audit rather than reading as a finding on
    a file that does not exist.
    """
    monkeypatch.setattr(
        Checkout,
        "tasks",
        property(lambda checkout: TypeAdapter(list[Task]).validate_python([{}]))
    )

    with raises(ValidationError):
        RunCheck(checkout=checkout).findings


@mark.parametrize(
    ("old", "new", "held"),
    [
        param(
            "-{{install_args_hash}}-",
            "-{{version}}-{{install_args_hash}}-",
            "{{version}}",
            id = "version"
        ),
        param(
            "-{{install_args_hash}}-",
            "-{{install_args_hash}}-{{file_hash}}-",
            "{{file_hash}}",
            id = "file-hash"
        ),
        param(
            "-{{install_args_hash}}-",
            "-{{install_args_hash}}-${{ github.run_id }}-",
            "${{ github.run_id }}",
            id = "expression"
        ),
        param(
            "mise-{{platform}}",
            "mise-{{cache_key_prefix}}-{{mise_env}}-{{bootstrap_hash}}-{{platform}}",
            None,
            id = "fixed"
        )
    ]
)
def test_a_cache_key_holding_a_value_that_can_change_is_named(
    checkout : Checkout,
    held     : str | None,
    new      : str,
    old      : str
):
    """
    Asserts that the `cache_key` the composite action passes the mise action
    is named for each value it holds before its `hashFiles` call that can
    change between saves of one tool subset, and that the variables fixed
    across those saves pass.
    """
    edit(checkout, ACTION, new=new, old=old)

    assert ProvisionCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(ACTION),
            message = f"The `cache_key` in `{ACTION}` holds `{held}` before its "
            "`hashFiles` call, whereas only a value fixed across the saves of one "
            "tool subset keeps `gha:prune` grouping them"
        )
    ] * bool(held)


@mark.parametrize(
    ("old", "new"),
    [
        param("mise.lock') }}", "mise.lock') }}-v2", id="suffixed"),
        param("{{install_args_hash}}-$", "{{install_args_hash}}$", id="joined"),
        param("cache_key    :", "cache_prefix :", id="missing")
    ]
)
def test_a_cache_key_ending_on_anything_but_its_hash_is_named(
    checkout : Checkout,
    new      : str,
    old      : str
):
    """
    Asserts that the composite action is named where the `cache_key` it
    passes the mise action ends on anything but a `-` and a `hashFiles`
    call, runs the call into the value before it, or is missing.
    """
    edit(checkout, ACTION, new=new, old=old)

    assert ProvisionCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(ACTION),
            message = f"`{ACTION}` passes `jdx/mise-action` no `cache_key` ending on a "
            "`-` and a `hashFiles` call, whose hash `gha:prune` drops to group the "
            "saves of one tool subset"
        )
    ]


def test_a_workflow_failing_to_read_is_reported_on_its_path(checkout: Checkout):
    """
    Asserts that a workflow yaml-rs cannot read, here one repeating a key,
    is reported as one finding on that workflow, whose error names no field.
    """
    (checkout.root / CI).write_text("name: one\nname: two\n")

    assert GateCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(CI),
            message = f"`{CI}`: Value error, duplicate mapping key: 'name'"
        )
    ]


def test_an_action_pinned_to_a_second_commit_is_named(checkout: Checkout):
    """
    Asserts that a step pinning an action to a commit other than the one
    the first step pinning it names is named against that first step, in
    path order.
    """
    edit(
        checkout = checkout,
        new = "      - name : Check out again\n        uses : actions/checkout@0f0f\n\n"
        "      - name : Cut the draft",
        old  = "      - name : Cut the draft",
        path = DRAFT
    )

    assert CommitCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(DRAFT),
            message = f"The pin on `actions/checkout` in `{DRAFT}` reads `0f0f`, where "
            f"the pin on it in `{ACTION}` reads "
            "`3d3c42e5aac5ba805825da76410c181273ba90b1`"
        )
    ]


def test_an_undeclared_input_to_a_composite_action_is_named(checkout: Checkout):
    """
    Asserts that a step passing a composite action an input its manifest
    does not declare is named on the workflow holding the step.
    """
    edit(
        checkout = checkout,
        new      = "with : { cache: 'false', tools: *package }",
        old      = "with : { tools: *package }",
        path     = CI
    )

    assert ProvisionCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(CI),
            message = f"`{CI}` passes `$/.github/actions/provision` the input `cache`, "
            f"which `{ACTION}` does not declare"
        )
    ]
