"""
Pins what each check `repo:audit` runs reports over a copy of the checkout,
nothing for the copy as it stands and one finding for each way a case breaks
it, the label registry's agreement with each file naming a label among them.
"""

from common.edits import append, edit, task
from pathlib      import Path
from pydantic     import TypeAdapter, ValidationError
from pytest       import MonkeyPatch, mark, param, raises
from shutil       import copystat

from thermur.repo.checkout import Checkout
from thermur.repo.checks   import (
    BinCheck, Check, LabelCheck, ParityCheck, PinCheck, RunCheck, WheelCheck
)
from thermur.repo.schemas  import Finding, Task

FLOOR = "the floor of `requires-python` in `pyproject.toml` reads `3.14`"
SHELL = "#!/usr/bin/env -S bash -euo pipefail\n"


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
            "given-names: James",
            "given-names: Jim",
            "The author list in `CITATION.cff` reads `Jim Parkington`, where the "
            "author list in `pyproject.toml` reads `James Parkington`",
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
    Asserts that each file restating the Python version, the uv release, the
    license, or what `CITATION.cff` cites is named once its copy drifts from
    the declaration it restates, or once the file stops carrying it.
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


def test_a_program_on_the_path_running_uv_run_unlocked_is_named(checkout: Checkout):
    """
    Asserts that a program under `.mise/bin/` is held to `--exact --locked`
    as a task is.
    """
    edit(checkout, ".mise/bin/pytest", new="", old="--exact --locked ")

    assert [finding.file for finding in RunCheck(checkout=checkout).findings] == [
        Path(".mise/bin/pytest")
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


@mark.parametrize(
    ("old", "new", "files"),
    [
        param(
            '">=3.14"',
            '">=3.15"',
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
    files    : list[str]
):
    """
    Asserts that raising `requires-python` names the README's badge, the
    formatter's `target-version`, and the minor mise pins, and that moving
    `[project].version` names `CITATION.cff`, each against the new value.
    """
    edit(checkout, "pyproject.toml", new=new, old=old)

    assert [finding.file for finding in ParityCheck(checkout=checkout).findings] == [
        Path(file) for file in files
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


def test_a_template_opening_on_no_front_matter_is_named(checkout: Checkout):
    """
    Asserts that a Markdown template whose front matter is gone, which
    GitHub leaves out of its template chooser, is named on its file.
    """
    edit(checkout, ".github/ISSUE_TEMPLATE/bug.md", new="", old="---\n")

    assert LabelCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(".github/ISSUE_TEMPLATE/bug.md"),
            message = "`.github/ISSUE_TEMPLATE/bug.md`: Input should be a valid "
            "dictionary or instance of Template"
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
