"""
Pins what each check `repo:audit` runs reports over a copy of the checkout,
nothing for the copy as it stands and one finding for each way a case breaks
it, the label registry's agreement with each file naming a label among them.
A case naming a pin reads its release through the `pins` fixture, so a lift
leaves every case as it stands.
"""

from common.edits import append, edit, task
from pathlib      import Path
from pydantic     import TypeAdapter, ValidationError
from pytest       import MonkeyPatch, mark, param, raises
from shutil       import copystat

from thermur.docs.outputs  import Outputs
from thermur.repo.checkout import Checkout
from thermur.repo.checks   import (
    AnchorCheck,
    BinCheck,
    Check,
    CommitCheck,
    FootnoteCheck,
    GateCheck,
    GuideCheck,
    ImageCheck,
    KitCheck,
    LabelCheck,
    LockfileCheck,
    ParityCheck,
    PinCheck,
    ProvisionCheck,
    RenderCheck,
    RulesetCheck,
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
FLOOR  = "the floor of `requires-python` in `pyproject.toml` reads `{floor}`"
MAIN   = ".github/rulesets/main.json"
SHELL  = "#!/usr/bin/env -S bash -euo pipefail\n"
SITE   = ".github/workflows/site.yml"
WARM   = ".github/workflows/warm.yml"


@mark.parametrize(
    "check",
    [FootnoteCheck, RenderCheck],
    ids = lambda check: check.__name__
)
def test_a_bibliography_failing_validation_is_named_on_its_path(
    check    : type[Check],
    checkout : Checkout
):
    """
    Asserts that an entry in `docs/references.bib` leaving out its priority
    is reported on the file by each check reading the bibliography, rather
    than raised.
    """
    edit(checkout, "docs/references.bib", new="", old="    priority = {prio1},\n")

    assert check(checkout=checkout).findings == [
        Finding(
            file    = Path("docs/references.bib"),
            message = "`studies.attanasi2014.article.priority` in "
            "`docs/references.bib`: Field required"
        )
    ]


def test_a_check_declaring_nothing_makes_no_finding(checkout: Checkout):
    """
    Asserts that the base check, comparing no restated declarations and
    scanning nothing of its own, makes no finding, the default each check
    holding pairs or a scan of its own builds on.
    """
    assert Check(checkout=checkout).findings == []


@mark.parametrize(
    ("path", "old", "new", "message"),
    [
        param(
            "README.md",
            "python-{floor}+",
            "python-3.13+",
            f"The Python badge in `README.md` reads `3.13`, where {FLOOR}",
            id = "python-badge"
        ),
        param(
            "README.md",
            "badge/python-{floor}+",
            "badge/py-3.14+",
            f"The Python badge in `README.md` is missing, where {FLOOR}",
            id = "python-badge-missing"
        ),
        param(
            "pyproject.toml",
            'target-version = "{floor}"',
            'target-version = "3.13"',
            f"The `target-version` in `[tool.prose]` reads `3.13`, where {FLOOR}",
            id = "target-version"
        ),
        param(
            ".mise/config.toml",
            'python                    = "{python}"',
            'python                    = "3.99.0"',
            "The minor of the `python` pin in `.mise/config.toml` reads `3.99`, "
            f"where {FLOOR}",
            id = "mise-python"
        ),
        param(
            "pyproject.toml",
            '"=={uv}"',
            '"==0.0.1"',
            "The `required-version` in `[tool.uv]` reads `==0.0.1`, where the "
            "`uv` pin in `.mise/config.toml` as a `==` requirement reads "
            "`=={uv}`",
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
            "Jane Doe",
            "The copyright holder list in `LICENSE` reads `Jane Doe`, where the author "
            "list in `pyproject.toml` reads `James Parkington`",
            id = "license-holders"
        ),
        param(
            "CITATION.cff",
            "version: 0.1.0",
            "version: 0.2.0",
            "The `version` in `CITATION.cff` reads `0.2.0`, where the `version` in "
            "`pyproject.toml` reads `0.1.0`",
            id = "citation-version"
        ),
        param(
            "CITATION.cff",
            "license: MIT",
            "license: Apache-2.0",
            "The `license` in `CITATION.cff` reads `Apache-2.0`, where the `license` "
            "in `pyproject.toml` reads `MIT`",
            id = "citation-license"
        ),
        param(
            "CITATION.cff",
            "repository-code: https://github.com/Jybbs/thermur",
            "repository-code: https://github.com/Jybbs/Thermur",
            "The `repository-code` in `CITATION.cff` reads "
            "`https://github.com/Jybbs/Thermur`, where the `Repository` in "
            "`pyproject.toml` reads `https://github.com/Jybbs/thermur`",
            id = "citation-repository"
        ),
        param(
            "CITATION.cff",
            "family-names: Parkington\n    given-names: James",
            "family-names: Doe\n    given-names: Jane",
            "The author list in `CITATION.cff` reads `Jane Doe`, where the author "
            "list in `pyproject.toml` reads `James Parkington`",
            id = "citation-authors"
        ),
        param(
            "CITATION.cff",
            "version: 0.1.0\n",
            "",
            "The `version` in `CITATION.cff` is missing, where the `version` in "
            "`pyproject.toml` reads `0.1.0`",
            id = "citation-version-missing"
        ),
        param(
            "CITATION.cff",
            "authors:\n  - family-names: Parkington\n    given-names: James\n",
            "",
            "The author list in `CITATION.cff` is missing, where the author list in "
            "`pyproject.toml` reads `James Parkington`",
            id = "citation-authors-missing"
        ),
        param(
            "docs/package.json",
            '"@vitest/coverage-v8" : "{vitest}"',
            '"@vitest/coverage-v8" : "0.0.1"',
            "The pin on `@vitest/coverage-v8` in `docs/package.json` reads `0.0.1`, "
            "where the pin on `vitest` there reads `{vitest}`",
            id = "coverage-provider"
        ),
        param(
            "docs/package.json",
            '"@types/node"         : "{types}"',
            '"@types/node"         : "1.0.0"',
            "The major of the pin on `@types/node` in `docs/package.json` reads `1`, "
            "where the major of the `node` pin in `.mise/config.toml` reads `{node}`",
            id = "node-types"
        )
    ]
)
def test_a_restated_declaration_that_drifts_is_named(
    checkout : Checkout,
    path     : str,
    old      : str,
    new      : str,
    message  : str,
    pins     : dict[str, str]
):
    """
    Asserts that each file restating the Python version, the uv release, the
    license, or what `CITATION.cff` cites is named once its copy drifts from
    the declaration it restates, or once the file stops carrying it.
    """
    edit(checkout, path, new=new, old=old.format_map(pins))

    assert ParityCheck(checkout=checkout).findings == [
        Finding(file=Path(path), message=message.format_map(pins))
    ]


def test_a_citation_naming_each_author_passes(checkout: Checkout):
    """
    Asserts that two authors, each written in `CITATION.cff` as a
    `family-names` line over a `given-names` line, read in order and join
    the way the manifest's authors do, so a citation, a manifest, and a
    copyright line naming the same two make no finding.
    """
    edit(
        checkout,
        "pyproject.toml",
        new = '"James Parkington" }, { name = "Ada Lovelace" }]',
        old = '"James Parkington" }]'
    )
    edit(
        checkout,
        "CITATION.cff",
        new = "    given-names: James\n  - family-names: Lovelace\n"
        "    given-names: Ada\n",
        old = "    given-names: James\n"
    )
    edit(
        checkout,
        "LICENSE",
        new = "James Parkington and Ada Lovelace",
        old = "James Parkington"
    )

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
        param(f"{SHELL}uv run\n--exact --locked pytest\n", True, id="next-line"),
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

    assert LockfileCheck(checkout=checkout).findings == [
        Finding(
            file    = path,
            message = f"`{path}` runs `uv run` without `--exact --locked` opening its "
            "arguments"
        )
    ] * named


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


def test_a_copyright_year_range_still_reads_its_holders(checkout: Checkout):
    """
    Asserts that a copyright line naming a range of years, as a license
    carried past its first year does, reads the holders after the range.
    """
    edit(checkout, "LICENSE", new="Copyright (c) 2025-2026", old="Copyright (c) 2025")

    assert ParityCheck(checkout=checkout).findings == []


def test_a_gate_running_under_a_bare_always_passes(checkout: Checkout):
    """
    Asserts that a gate whose condition is `always()` written without the
    `${{ }}` GitHub lets a condition leave out passes.
    """
    edit(checkout, CI, new="if      : always()", old="if      : ${{ always() }}")

    assert GateCheck(checkout=checkout).findings == []


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


def test_a_guide_without_its_checks_table_lists_no_check(checkout: Checkout):
    """
    Asserts that a contributor guide whose checks table loses its header
    reads as listing no check, so every row of a pull-request workflow is
    named as missing from it.
    """
    edit(
        checkout,
        ".github/CONTRIBUTING.md",
        new = "| **Name** | **Runs Locally** |",
        old = "| **Check** | **Runs Locally** |"
    )

    commands = {task.name: task.command for task in checkout.tasks}

    assert [finding.message for finding in GuideCheck(checkout=checkout).findings] == [
        f"The command beside `{name}` in `.github/CONTRIBUTING.md` is missing, "
        f"where the command running `{task}` reads `{commands[task]}`"
        for name, task in checkout.checks
    ]


@mark.parametrize(
    ("old", "new", "message"),
    [
        param(
            '"hatchling=={hatchling}"',
            '"hatchling>={hatchling}"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-range"
        ),
        param(
            '"hatchling=={hatchling}"',
            '"hatchling==1.*"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-wildcard"
        ),
        param(
            '"hatchling=={hatchling}"',
            '"setuptools==80.9.0"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-absent"
        ),
        param(
            '"pluggy=={pluggy}"',
            '"pluggy>=1,<99"',
            "The build constraint `pluggy<99,>=1` in `[tool.uv]` is not an exact pin",
            id = "constraint-range"
        ),
        param(
            '"hatchling=={hatchling}"',
            '"hatchling"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-unpinned"
        ),
        param(
            '"pluggy=={pluggy}"',
            '"pluggy"',
            "The build constraint `pluggy` in `[tool.uv]` is not an exact pin",
            id = "constraint-unpinned"
        ),
        param(
            '"hatchling=={hatchling}"',
            '"hatchling=={hatchling},<99"',
            "The `requires` of `[build-system]` holds no exact pin on hatchling",
            id = "hatchling-second-specifier"
        )
    ]
)
def test_a_build_requirement_naming_a_range_is_named(
    checkout : Checkout,
    old      : str,
    new      : str,
    message  : str,
    pins     : dict[str, str]
):
    """
    Asserts that a hatchling requirement naming a range, a wildcard, or no
    hatchling at all, and a build constraint naming a range, are each named,
    since uv resolves both outside `uv.lock`.
    """
    edit(checkout, "pyproject.toml", new=new.format_map(pins), old=old.format_map(pins))

    assert PinCheck(checkout=checkout).findings == [
        Finding(file=Path("pyproject.toml"), message=message)
    ]


@mark.parametrize(
    ("path", "old", "new", "messages"),
    [
        param(
            "pyproject.toml",
            'requires-python = ">={floor}"\n',
            "",
            ["`project.requires-python` in `pyproject.toml`: Field required"],
            id = "missing-field"
        ),
        param(
            "pyproject.toml",
            'license         = "MIT"\nname            = "thermur"\n'
            'readme          = "README.md"\nrequires-python = ">={floor}"\n',
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
            '"hatchling=={hatchling}"',
            '"hatchling={hatchling}"',
            [
                "`build-system.requires.0` in `pyproject.toml`: Value error, Expected "
                "semicolon (after name with no version specifier) or end\n    "
                "hatchling={hatchling}\n             ^"
            ],
            id = "malformed-requirement"
        ),
        param(
            ".mise/config.toml",
            'python                    = "{python}"',
            'python                    = "3"',
            [
                "`tools.python` in `.mise/config.toml`: Value error, names no major "
                "and minor version"
            ],
            id = "unversioned-python"
        ),
        param(
            "CITATION.cff",
            "  - family-names: Parkington\n    given-names: James\n",
            "",
            ["`authors` in `CITATION.cff`: Input should be a valid list"],
            id = "citation-authors-empty"
        )
    ]
)
def test_a_document_failing_validation_is_reported_rather_than_raised(
    checkout : Checkout,
    path     : str,
    old      : str,
    new      : str,
    messages : list[str],
    pins     : dict[str, str]
):
    """
    Asserts that a document missing fields the checks read, or holding one
    that fails to parse, makes one finding on that document per field rather
    than ending the audit, whichever field the check itself reads.
    """
    edit(checkout, path, new=new.format_map(pins), old=old.format_map(pins))

    assert ParityCheck(checkout=checkout).findings == [
        Finding(file=Path(path), message=message.format_map(pins))
        for message in messages
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
        param('["uv run pytest", { task = "py:check" }]', id="beside-a-reference"),
        param("'''uv \\\nrun pytest'''", id="continued")
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
    failing validation, and a `uv run` a backslash continues across lines.
    """
    append(checkout, ".mise/config.toml", f'\n[tasks."py:sweep"]\nrun = {run}\n')

    assert [finding.file for finding in LockfileCheck(checkout=checkout).findings] == [
        Path(".mise/config.toml")
    ]


def test_a_guide_without_its_label_table_describes_no_label(checkout: Checkout):
    """
    Asserts that a contributor guide whose label table loses its header
    reads as describing no label, so every label the registry declares is
    named as missing from it.
    """
    edit(
        checkout,
        ".github/CONTRIBUTING.md",
        new = "| **Name** | **Covers** |",
        old = "| **Label** | **Covers** |"
    )

    assert [finding.message for finding in LabelCheck(checkout=checkout).findings] == [
        f"The description of `{label.name}` in `.github/CONTRIBUTING.md` is missing, "
        f"where its description in `.github/labels.toml` reads `{label.description}`"
        for label in checkout.labels.rows
    ]


def test_a_hatchling_pin_in_another_spelling_passes(
    checkout : Checkout,
    pins     : dict[str, str]
):
    """
    Asserts that an exact pin naming hatchling in another spelling its
    normalized name allows passes, since a package's name compares case
    insensitively.
    """
    edit(
        checkout,
        "pyproject.toml",
        new = f'"Hatchling=={pins["hatchling"]}"',
        old = f'"hatchling=={pins["hatchling"]}"'
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


def test_a_marker_naming_no_entry_is_named_at_its_line(checkout: Checkout):
    """
    Asserts that a marker under a study's key the bibliography lacks is
    named at its line on the page carrying it, whereas a numbered marker on
    the same page is not.
    """
    page = Path("docs/pages/notes.md")
    (checkout.root / page).write_text(
        "# Notes\n\nA note[^1].\n\nA study[^smith2020].\n"
    )

    assert FootnoteCheck(checkout=checkout).findings == [
        Finding(
            file    = page,
            message = f"Line 5 of `{page}` cites `smith2020`, which names no entry in "
            "`docs/references.bib`"
        )
    ]


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


def test_a_page_whose_definitions_drift_is_named(checkout: Checkout):
    """
    Asserts that a page citing a study whose definition it leaves out is
    named as drifted from the bibliography.
    """
    append(checkout, "README.md", "\nOrdered flocks[^cavagna2010].\n")

    assert RenderCheck(checkout=checkout).findings == [
        Finding(
            file    = Path("README.md"),
            message = "The footnote definitions in `README.md` have drifted from their "
            "entries in `docs/references.bib`, and `mise render` writes them again"
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


def test_a_program_on_the_path_running_uv_run_unlocked_is_named(checkout: Checkout):
    """
    Asserts that a program under `.mise/bin/` is held to `--exact --locked`
    as a task is.
    """
    edit(checkout, ".mise/bin/pytest", new="", old="--exact --locked ")

    assert [finding.file for finding in LockfileCheck(checkout=checkout).findings] == [
        Path(".mise/bin/pytest")
    ]


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
            "minijinja==0.0.1",
            f"The pin on `minijinja` in `{BRIEF}` reads `==0.0.1`, where the version "
            "`uv.lock` resolves reads `=={minijinja}`",
            id = "drifted"
        ),
        param(
            "minijinja",
            f"The pin on `minijinja` in `{BRIEF}` is missing, where the version "
            "`uv.lock` resolves reads `=={minijinja}`",
            id = "unpinned"
        ),
        param("rich==14.0.0", None, id="unlocked")
    ]
)
def test_a_script_pinning_another_version_than_the_lock_is_named(
    checkout : Checkout,
    message  : str | None,
    new      : str,
    pins     : dict[str, str]
):
    """
    Asserts that a task script pinning another version of a package
    `uv.lock` resolves, or pinning none, is named, and that one requiring a
    package `uv.lock` does not resolve passes.
    """
    edit(checkout, BRIEF, new=new, old=f"minijinja=={pins['minijinja']}")

    assert [finding.message for finding in ScriptCheck(checkout=checkout).findings] == (
        [message.format_map(pins)] if message else []
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

    assert LockfileCheck(checkout=checkout).findings == [
        Finding(
            file    = path,
            message = f"`{path}` runs `uv run` without `--exact --locked` opening its "
            "arguments"
        )
    ]


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


def test_a_renamed_label_is_named_in_every_file_naming_the_old_one(checkout: Checkout):
    """
    Asserts that renaming a label in the registry alone names the
    release-notes categories, the bug and spec templates in that order, and
    the guide, each still naming the old label, then the new label filed
    under no category and missing from the guide.
    """
    edit(checkout, ".github/ISSUE_TEMPLATE/spec.md", new='["🐞 bug"]', old="[]")
    edit(checkout, ".github/labels.toml", new='"🐛 bug"', old='"🐞 bug"')

    assert [finding.file for finding in LabelCheck(checkout=checkout).findings] == [
        Path(".github/release.yml"),
        Path(".github/ISSUE_TEMPLATE/bug.md"),
        Path(".github/ISSUE_TEMPLATE/spec.md"),
        Path(".github/CONTRIBUTING.md"),
        Path(".github/release.yml"),
        Path(".github/CONTRIBUTING.md")
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


def test_a_second_pull_request_gate_keeps_the_required_check(checkout: Checkout):
    """
    Asserts that renaming the gate of one workflow running on a pull request
    makes no finding while another still ends on `📡 Readout`, since the
    audit reads the gates of every such workflow as one set.
    """
    edit(checkout, CI, new="name    : 📡 Report", old="name    : 📡 Readout")

    assert RulesetCheck(checkout=checkout).findings == []


@mark.parametrize(
    ("path", "old", "new", "findings"),
    [
        param(
            ".github/release.yml",
            '["🐞 bug"]',
            '["🐛 bug"]',
            [
                (
                    ".github/release.yml",
                    "`.github/release.yml` names `🐛 bug`, a label "
                    "`.github/labels.toml` does not declare"
                ),
                (
                    ".github/release.yml",
                    "`.github/release.yml` files `🐞 bug` under no release-notes "
                    "categories rather than one"
                )
            ],
            id = "release-undeclared"
        ),
        param(
            ".github/release.yml",
            '["🦜 cli"]',
            '["🦜 cli", "🐞 bug"]',
            [
                (
                    ".github/release.yml",
                    "`.github/release.yml` files `🐞 bug` under 2 release-notes "
                    "categories rather than one"
                )
            ],
            id = "release-twice"
        ),
        param(
            ".github/ISSUE_TEMPLATE/bug.md",
            '["🐞 bug"]',
            '["🐞bug"]',
            [
                (
                    ".github/ISSUE_TEMPLATE/bug.md",
                    "`.github/ISSUE_TEMPLATE/bug.md` names `🐞bug`, a label "
                    "`.github/labels.toml` does not declare"
                )
            ],
            id = "template-undeclared"
        ),
        param(
            ".github/CONTRIBUTING.md",
            "| `🦜 cli` | The command line |",
            "| `🦜 cli` | The commands |",
            [
                (
                    ".github/CONTRIBUTING.md",
                    "The description of `🦜 cli` in `.github/CONTRIBUTING.md` reads "
                    "`The commands`, where its description in `.github/labels.toml` "
                    "reads `The command line`"
                )
            ],
            id = "guide-description"
        ),
        param(
            ".github/CONTRIBUTING.md",
            "| `🦜 cli` | The command line |\n",
            "",
            [
                (
                    ".github/CONTRIBUTING.md",
                    "The description of `🦜 cli` in `.github/CONTRIBUTING.md` is "
                    "missing, where its description in `.github/labels.toml` reads "
                    "`The command line`"
                )
            ],
            id = "guide-missing"
        ),
        param(
            ".github/CONTRIBUTING.md",
            "| `🦜 cli` | The command line |\n",
            "| `🦜 cli` | The command line |\n| `✨ feature` | A new capability |\n",
            [
                (
                    ".github/CONTRIBUTING.md",
                    "`.github/CONTRIBUTING.md` names `✨ feature`, a label "
                    "`.github/labels.toml` does not declare"
                )
            ],
            id = "guide-undeclared"
        ),
        param(
            ".github/labels.toml",
            'color       = "8c055e"',
            'color       = "8C055E"',
            [
                (
                    ".github/labels.toml",
                    "`labels.0.color` in `.github/labels.toml`: String should match "
                    "pattern '^[0-9a-f]{6}$'"
                )
            ],
            id = "registry-malformed"
        ),
        param(
            ".github/labels.toml",
            'description = "Wrong output or a broken run in any domain"',
            f'description = "{"x" * 101}"',
            [
                (
                    ".github/labels.toml",
                    "`labels.0.description` in `.github/labels.toml`: String should "
                    "have at most 100 characters"
                )
            ],
            id = "registry-overlong-description"
        ),
        param(
            ".github/release.yml",
            '- labels: ["🐞 bug"]',
            "- labels:\n        - 🐛 bug",
            [
                (
                    ".github/release.yml",
                    "`.github/release.yml` names `🐛 bug`, a label "
                    "`.github/labels.toml` does not declare"
                ),
                (
                    ".github/release.yml",
                    "`.github/release.yml` files `🐞 bug` under no release-notes "
                    "categories rather than one"
                )
            ],
            id = "release-block-list"
        ),
        param(
            ".github/release.yml",
            '["🦜 cli"]',
            '["🐛 bug", "🐛 bug"]',
            [
                (
                    ".github/release.yml",
                    "`.github/release.yml` names `🐛 bug`, a label "
                    "`.github/labels.toml` does not declare"
                ),
                (
                    ".github/release.yml",
                    "`.github/release.yml` files `🦜 cli` under no release-notes "
                    "categories rather than one"
                )
            ],
            id = "release-repeated"
        ),
        param(
            ".github/ISSUE_TEMPLATE/bug.md",
            'labels: ["🐞 bug"]',
            "labels: 🐛 bug, 🔬 discovery",
            [
                (
                    ".github/ISSUE_TEMPLATE/bug.md",
                    "`.github/ISSUE_TEMPLATE/bug.md` names `🐛 bug`, a label "
                    "`.github/labels.toml` does not declare"
                )
            ],
            id = "template-comma-delimited"
        ),
        param(
            ".github/ISSUE_TEMPLATE/bug.md",
            'labels: ["🐞 bug"]',
            "labels: ['🐛 bug']",
            [
                (
                    ".github/ISSUE_TEMPLATE/bug.md",
                    "`.github/ISSUE_TEMPLATE/bug.md` names `🐛 bug`, a label "
                    "`.github/labels.toml` does not declare"
                )
            ],
            id = "template-single-quoted"
        ),
        param(
            ".github/ISSUE_TEMPLATE/bug.md",
            'labels: ["🐞 bug"]',
            "labels:\n  - 🐛 bug",
            [
                (
                    ".github/ISSUE_TEMPLATE/bug.md",
                    "`.github/ISSUE_TEMPLATE/bug.md` names `🐛 bug`, a label "
                    "`.github/labels.toml` does not declare"
                )
            ],
            id = "template-block-list"
        ),
        param(
            ".github/CONTRIBUTING.md",
            "| `🧰 tooling` | The packaging, dependencies, workflows, and release |\n",
            "| `🧰 tooling` | The packaging, dependencies, workflows, and release |\n"
            "\nA paragraph.\n\n| `✨ feature` | A new capability |\n",
            [],
            id = "guide-later-table"
        )
    ]
)
def test_a_file_disagreeing_with_the_label_registry_is_named(
    checkout : Checkout,
    path     : str,
    old      : str,
    new      : str,
    findings : list[tuple[str, str]]
):
    """
    Asserts that a release-notes category or an issue template naming a
    label the registry does not declare, in any form YAML writes a list or
    a template writes one string of names, a declared label filed under no
    category or under two, a guide describing a label otherwise or naming
    one the registry lacks, and a registry failing its own validation are
    each named on the file a reader opens to put it right.
    """
    edit(checkout, path, new=new, old=old)

    assert LabelCheck(checkout=checkout).findings == [
        Finding(file=Path(file), message=message) for file, message in findings
    ]


@mark.parametrize(
    ("old", "new", "files"),
    [
        param(
            '">={floor}"',
            '">=3.99"',
            ["README.md", "pyproject.toml", ".mise/config.toml"],
            id = "python-floor"
        ),
        param(
            'version         = "0.1.0"',
            'version         = "0.2.0"',
            ["CITATION.cff"],
            id = "version"
        )
    ]
)
def test_a_moved_manifest_declaration_names_every_copy(
    checkout : Checkout,
    old      : str,
    new      : str,
    files    : list[str],
    pins     : dict[str, str]
):
    """
    Asserts that raising `requires-python` names the README's badge, the
    formatter's `target-version`, and the minor mise pins, and that moving
    `[project].version` names `CITATION.cff`, each against the new value.
    """
    edit(checkout, "pyproject.toml", new=new, old=old.format_map(pins))

    assert [finding.file for finding in ParityCheck(checkout=checkout).findings] == [
        Path(file) for file in files
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


def test_a_template_opening_on_no_front_matter_is_named(checkout: Checkout):
    """
    Asserts that a Markdown template opening on no front matter is named on
    its file rather than read as naming no label.
    """
    edit(checkout, ".github/ISSUE_TEMPLATE/bug.md", new="", old="---\n")

    assert LabelCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(".github/ISSUE_TEMPLATE/bug.md"),
            message = "`.github/ISSUE_TEMPLATE/bug.md`: Value error, opens on no "
            "front matter"
        )
    ]


@mark.parametrize(
    ("name", "text", "findings"),
    [
        param(
            "feature.yml",
            'name: Feature\nlabels: ["✨ feature"]\n',
            [
                Finding(
                    file    = Path(".github/ISSUE_TEMPLATE/feature.yml"),
                    message = "`.github/ISSUE_TEMPLATE/feature.yml` names "
                    "`✨ feature`, a label `.github/labels.toml` does not declare"
                )
            ],
            id = "issue-form"
        ),
        param("config.yml", "blank_issues_enabled: false\n", [], id="chooser")
    ]
)
def test_a_file_added_under_the_issue_templates_is_read(
    checkout : Checkout,
    name     : str,
    text     : str,
    findings : list[Finding]
):
    """
    Asserts that an issue form under `.github/ISSUE_TEMPLATE/`, read beside
    the Markdown templates, is named where it lists a label the registry
    does not declare, whereas the template chooser's configuration, which
    lists no labels, names none.
    """
    (checkout.root / ".github/ISSUE_TEMPLATE" / name).write_text(text)

    assert LabelCheck(checkout=checkout).findings == findings


@mark.parametrize(
    ("check", "path", "old", "new"),
    [
        param(ParityCheck, "pyproject.toml", "[project]", "[project", id="toml"),
        param(LabelCheck, ".github/release.yml", '["*"]', '["*"', id="yaml"),
        param(
            LabelCheck,
            ".github/ISSUE_TEMPLATE/bug.md",
            'labels: ["🐞 bug"]',
            'labels: ["🐞 bug"',
            id = "front-matter"
        ),
        param(ParityCheck, "CITATION.cff", "license: MIT", "license: [MIT", id="cff")
    ]
)
def test_a_document_failing_to_parse_is_named_on_its_file(
    check    : type[Check],
    checkout : Checkout,
    path     : str,
    old      : str,
    new      : str
):
    """
    Asserts that a document whose TOML or YAML fails to parse, a Markdown
    template's front matter among them, makes one finding on that document
    carrying the parser's error rather than ending the audit.
    """
    edit(checkout, path, new=new, old=old)

    assert [
        (finding.file, finding.message.startswith(f"`{path}`: Value error, "))
        for finding in check(checkout=checkout).findings
    ] == [(Path(path), True)]


@mark.parametrize(
    ("old", "new"),
    [
        param("version: 0.1.0", 'version: "0.1.0"', id="version"),
        param("license: MIT", "license: 'MIT'", id="license"),
        param(
            "repository-code: https://github.com/Jybbs/thermur",
            'repository-code: "https://github.com/Jybbs/thermur"',
            id = "repository-code"
        ),
        param("given-names: James", 'given-names: "James"', id="authors")
    ]
)
def test_a_quoted_citation_value_reads_as_the_value_it_quotes(
    checkout : Checkout,
    old      : str,
    new      : str
):
    """
    Asserts that a `CITATION.cff` value written in single or double quotes,
    which YAML reads as the same string, agrees with the manifest as the
    plain value does.
    """
    edit(checkout, "CITATION.cff", new=new, old=old)

    assert ParityCheck(checkout=checkout).findings == []


@mark.parametrize(
    ("path", "old", "new", "messages"),
    [
        param(
            ".github/CONTRIBUTING.md",
            "| `🪵 Lockfiles` | `mise lockfile` |",
            "| `🪵 Lockfiles` | `mise run lock:check` |",
            [
                "The command beside `🪵 Lockfiles` in `.github/CONTRIBUTING.md` reads "
                "`mise run lock:check`, where the command running "
                "`lock:check` reads `mise lockfile`"
            ],
            id = "command"
        ),
        param(
            CI,
            "name  : 🪵 Lockfiles",
            "name  : 🪵 Locks",
            [
                "`.github/CONTRIBUTING.md` lists `🪵 Lockfiles`, a check no row of a "
                "pull-request workflow runs",
                "The command beside `🪵 Locks` in `.github/CONTRIBUTING.md` is "
                "missing, where the command running `lock:check` reads "
                "`mise lockfile`"
            ],
            id = "renamed-row"
        ),
        param(
            CI,
            "task  : repo:vet",
            "task  : gha:brief",
            [
                "The command beside `🗝️ Workflows` in `.github/CONTRIBUTING.md` "
                "reads `mise vet`, where the command running `gha:brief` "
                "reads `mise run gha:brief`"
            ],
            id = "task-without-alias"
        ),
        param(
            CI,
            "task  : repo:vet",
            "task  : repo:gone",
            [
                "The command beside `🗝️ Workflows` in `.github/CONTRIBUTING.md` "
                "reads `mise vet`, where the command running `repo:gone` "
                "reads `mise run repo:gone`"
            ],
            id = "unlisted-task"
        ),
        param(
            ".github/CONTRIBUTING.md",
            "| `🪵 Lockfiles` | `mise lockfile` |",
            "| `🪵 Lockfiles` | x | x |\n| `🪵 Lockfiles` | `mise lockfile` |",
            [
                "`checks` in `.github/CONTRIBUTING.md`: Value error, names "
                "`🪵 Lockfiles` more than once"
            ],
            id = "repeated-row"
        )
    ]
)
def test_a_guide_out_of_step_with_the_rows_is_named(
    checkout : Checkout,
    messages : list[str],
    new      : str,
    old      : str,
    path     : str
):
    """
    Asserts that the contributor guide giving a row's check a command other
    than the one running its task, whether or not mise lists that task or it
    declares an alias, the guide listing a check no row runs while leaving
    out one a row does, and the guide listing one check in two rows are each
    named on the guide.
    """
    edit(checkout, path, new=new, old=old)

    assert GuideCheck(checkout=checkout).findings == [
        Finding(file=Path(".github/CONTRIBUTING.md"), message=message)
        for message in messages
    ]


def test_a_toml_task_continuing_uv_run_onto_its_flags_passes(checkout: Checkout):
    """
    Asserts that a task declared in TOML passes where its `uv run` takes
    `--exact --locked` on the line a backslash continues it onto.
    """
    append(
        checkout,
        ".mise/config.toml",
        "\n[tasks.\"py:sweep\"]\nrun = '''uv run \\\n--exact --locked pytest'''\n"
    )

    assert LockfileCheck(checkout=checkout).findings == []


@mark.parametrize(
    ("paths", "old", "new", "context"),
    [
        param(
            (MAIN,),
            '"context": "📡 Readout"',
            '"context": "🐦‍⬛ CI"',
            "🐦‍⬛ CI",
            id = "required"
        ),
        param(
            (CI, SITE),
            "name    : 📡 Readout",
            "name    : 📡 Report",
            "📡 Readout",
            id = "gate"
        ),
        param(
            (MAIN,),
            '"integration_id": 15368 }]',
            '"integration_id": 15368 }, { "context": "🐦‍⬛ CI" }]',
            "🐦‍⬛ CI",
            id = "second"
        ),
        param((CI, SITE), "  gate:\n", "  brief:\n", "📡 Readout", id="no-gate")
    ]
)
def test_a_required_check_no_pull_request_gate_names_is_named(
    checkout : Checkout,
    context  : str,
    new      : str,
    old      : str,
    paths    : tuple[str, ...]
):
    """
    Asserts that renaming the check the `main` ruleset requires, adding a
    second required check beside it, renaming the gate of every workflow
    running on a pull request, or ending each of them on no gate makes one
    finding on the ruleset naming the required check no such gate carries,
    although the gates of the push workflows still carry `📡 Readout`.
    """
    for file in paths:
        edit(checkout, file, new=new, old=old)

    assert RulesetCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(MAIN),
            message = f"`{MAIN}` requires the `{context}` check, which no gate of a "
            "workflow running on a pull request names"
        )
    ]


def test_a_tools_input_to_another_action_installs_no_subset(
    checkout : Checkout,
    pins     : dict[str, str]
):
    """
    Asserts that a `tools` input a step passes an action other than a
    composite action of the checkout names no subset a Kit job must save.
    """
    edit(
        checkout = checkout,
        new      = f"uses : actions/checkout@{pins['checkout']}\n"
        "        with : { tools: bun }",
        old  = "uses : $/.github/actions/provision\n        with : { depth: '2' }",
        path = DRAFT
    )

    assert KitCheck(checkout=checkout).findings == []


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
        LockfileCheck(checkout=checkout).findings


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


def test_a_written_file_drifting_from_its_declaration_is_named(checkout: Checkout):
    """
    Asserts that a copy of the identity edited by hand, rather than written
    by the render task from the package's declaration, is named on its path.
    """
    path = Outputs().identity
    edit(checkout, str(path), new='"alpha": 0.5', old='"alpha": 0.64')

    assert RenderCheck(checkout=checkout).findings == [
        Finding(
            file    = path,
            message = f"`{path}` has drifted from the declaration it is written from, "
            "which `mise render` writes again"
        )
    ]


@mark.parametrize(
    ("text", "messages"),
    [
        param(f"{SHELL}bun install\n", ["install"], id="bare"),
        param(f"{SHELL}bunx vitepress build\n", ["bunx"], id="bunx"),
        param(f"{SHELL}bun install\nbunx knip\n", ["install", "bunx"], id="both"),
        param(f"{SHELL}bun install --frozen-lockfile\n", [], id="frozen"),
        param(
            f"{SHELL}bun install --cwd docs --lockfile-only\n",
            [],
            id = "lockfile-only"
        ),
        param(
            f"{SHELL}bun install \\\n  --frozen-lockfile \\\n  --ignore-scripts\n",
            [],
            id = "continued"
        ),
        param(f"{SHELL}# bunx is never run\nbun audit\n", [], id="comment")
    ]
)
def test_a_task_installing_off_the_lockfile_or_running_bunx_is_named(
    checkout : Checkout,
    messages : list[str],
    text     : str
):
    """
    Asserts that a task running `bun install` with neither
    `--frozen-lockfile` nor `--lockfile-only` is named, as is one running
    `bunx`, whereas one carrying the flag on a line a backslash continues
    the install onto passes, as does one naming `bunx` only in a comment.
    """
    path    = task(checkout, "site/fetch", text)
    written = {
        "bunx": f"`{path}` runs `bunx`, which can run a copy of a tool from outside "
        "`docs/node_modules`",

        "install": f"`{path}` runs `bun install` with neither `--frozen-lockfile` nor "
        "`--lockfile-only`"
    }

    assert LockfileCheck(checkout=checkout).findings == [
        Finding(file=path, message=written[message]) for message in messages
    ]


def test_an_action_pinned_to_a_second_commit_is_named(
    checkout : Checkout,
    pins     : dict[str, str]
):
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
            f"the pin on it in `{ACTION}` reads `{pins['checkout']}`"
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
