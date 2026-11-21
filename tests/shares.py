"""
Pins the `--share` option `tests/conftest.py` adds, which splits the suite
by test across several machines, meaning the tests each share runs under any
`pytest-randomly` seed, the unused snapshot every share reports, and the one
entry for each source file in the coverage data combined from every share.
"""

from coverage       import Coverage, CoverageData
from os             import pathsep
from pathlib        import Path
from pytest         import Config, MonkeyPatch, fixture, mark, param, raises
from subprocess     import CompletedProcess, run
from sys            import executable
from tests.conftest import share

CASES = [f"test_{number}" for number in range(5)]


@fixture
def suite(monkeypatch: MonkeyPatch, pytestconfig: Config, tmp_path: Path) -> Path:
    """
    Writes a suite under `tmp_path` whose conftest imports the share hooks
    from `tests/conftest.py`, holding one module of the cases `CASES` names,
    and records the snapshot each case matches. It clears the variable
    naming the pytest-xdist worker running this case, under which syrupy
    reads the suite's own run as a worker and leaves its unused snapshots to
    a controller that never runs.

    Returns:
        The path of the module holding the cases.
    """
    root = pytestconfig.rootpath
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("PYTHONPATH", pathsep.join(map(str, [root, root / "tests"])))
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import pytest_addoption, pytest_collection_modifyitems\n",
        encoding = "utf-8"
    )
    module = tmp_path / "test_cases.py"
    written(CASES, module)
    ran(tmp_path, "--snapshot-update", check=True)
    return module


def ran(folder: Path, *options: str, check: bool = False) -> CompletedProcess[str]:
    """
    Runs pytest over `folder` on one process with `options`, listing each
    test's outcome.

    Returns:
        The finished run, its output captured.
    """
    return run(
        [
            executable, "-m", "pytest", "--numprocesses=0",
            "-p", "no:cacheprovider", "-v",
            *options,
            str(folder)
        ],
        capture_output = True,
        check          = check,
        text           = True
    )


@mark.parametrize(
    "seeds",
    [param((1, 1), id="one-seed"), param((1, 2), id="two-seeds")]
)
def test_the_shares_run_every_test_once_under_any_seed(
    seeds : tuple[int, int],
    suite : Path
):
    """
    Asserts that the two shares of the suite pass every case between them,
    each case in one share alone, whether each share runs under the same
    `pytest-randomly` seed or under its own.
    """
    passed = [
        {
            case
            for case in CASES
            if f"test_cases.py::{case} PASSED" in ran(
                suite.parent,
                f"--randomly-seed={seed}",
                f"--share={index}/2"
            ).stdout
        }
        for index, seed in enumerate(seeds, start=1)
    ]

    assert (passed[0] | passed[1], passed[0] & passed[1]) == (set(CASES), set())


@mark.parametrize(
    ("kept", "status"),
    [
        param(CASES, 0, id="every-snapshot-read"),
        param(CASES[1:], 1, id="one-snapshot-unread")
    ]
)
@mark.parametrize("index", [1, 2])
def test_every_share_reports_a_snapshot_no_test_reads(
    index  : int,
    kept   : list[str],
    status : int,
    suite  : Path
):
    """
    Asserts that each share fails on a snapshot whose case the module
    no longer holds, and passes where every snapshot belongs to a case,
    including the snapshots of the cases another share runs.
    """
    written(kept, suite)

    assert ran(suite.parent, f"--share={index}/2").returncode == status


def test_the_shares_combine_into_one_entry_for_each_source_file(
    pytestconfig : Config,
    tmp_path     : Path
):
    """
    Asserts that coverage data measured in two checkouts at different paths
    under the `pyproject.toml` coverage settings combines into one entry for
    a source file both measured, which their relative paths make possible.
    """
    manifest  = pytestconfig.rootpath / "pyproject.toml"
    data_file = Coverage(config_file=manifest).config.data_file
    settings  = f"--rcfile={manifest}"
    checkouts = [tmp_path / name for name in ("one", "two")]
    for checkout in checkouts:
        (checkout / "src/thermur").mkdir(parents=True)
        (checkout / "src/thermur/flock.py").write_text(
            "flown = True\n",
            encoding = "utf-8"
        )
        run(
            [executable, "-m", "coverage", "run", settings, "src/thermur/flock.py"],
            capture_output = True,
            check          = True,
            cwd            = checkout
        )
    run(
        [
            executable, "-m", "coverage", "combine", settings,
            *(str(checkout / data_file) for checkout in checkouts)
        ],
        capture_output = True,
        check          = True,
        cwd            = tmp_path
    )
    combined = CoverageData(basename=tmp_path / data_file)
    combined.read()

    assert sorted(combined.measured_files()) == ["src/thermur/flock.py"]


@mark.parametrize("text", ["0/2", "3/2", "1", "one/two"])
def test_a_share_outside_its_count_is_refused(text: str):
    """
    Asserts that `share` refuses an index outside 1 to the count and text
    holding no pair of whole numbers.
    """
    with raises(ValueError):
        share(text)


def written(cases: list[str], module: Path):
    """
    Writes into `module` one case for each name in `cases`, each matching a
    snapshot of its own name.
    """
    module.write_text(
        "".join(
            f"def {case}(snapshot):\n    assert snapshot == {case!r}\n\n"
            for case in cases
        ),
        encoding = "utf-8"
    )
