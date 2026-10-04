"""
Pins what each check `repo:audit` runs reports over a copy of the checkout,
nothing for the copy as it stands and one finding for each way a case breaks
it.
"""

from common.edits import append, edit, task
from pathlib      import Path
from pydantic     import TypeAdapter, ValidationError
from pytest       import MonkeyPatch, mark, param, raises

from thermur.repo.checkout import Checkout
from thermur.repo.checks   import (
    BinCheck, Check, ParityCheck, PinCheck, RunCheck, WheelCheck
)
from thermur.repo.schemas  import Finding, Task

FLOOR = "the floor of `requires-python` in `pyproject.toml` reads `3.14`"
SHELL = "#!/usr/bin/env -S bash -euo pipefail\n"


def test_a_copy_differing_byte_for_byte_is_named(checkout: Checkout):
    """
    Asserts that a program whose bytes differ from the first program's is
    named against that first program.
    """
    append(checkout, ".mise/bin/pytest", "\n")

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


def test_a_linked_program_is_named(checkout: Checkout):
    """
    Asserts that a program written as a symlink to another is named, and is
    left out of the byte comparison it would otherwise pass.
    """
    linked = checkout.root / ".mise/bin/pytest"
    linked.unlink()
    linked.symlink_to("prose")

    assert BinCheck(checkout=checkout).findings == [
        Finding(
            file    = Path(".mise/bin/pytest"),
            message = "`.mise/bin/pytest` is a symlink rather than a copy"
        )
    ]


@mark.parametrize(
    ("text", "named"),
    [
        param(f"{SHELL}uv run pytest\n", True, id="bare"),
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

    assert bool(WheelCheck(checkout=checkout).findings) is named


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


def test_a_toml_task_running_uv_run_unlocked_is_named_on_its_configuration(
    checkout: Checkout
):
    """
    Asserts that a task `.mise/config.toml` declares in TOML is held to the
    lines it runs and named on that configuration.
    """
    append(
        checkout,
        ".mise/config.toml",
        '\n[tasks."py:sweep"]\nrun = "uv run pytest"\n'
    )

    assert [finding.file for finding in RunCheck(checkout=checkout).findings] == [
        Path(".mise/config.toml")
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
