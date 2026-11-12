"""
Pins the commands `thermur cut` runs, meaning that `thermur cut sample` cuts
the pinned sample from the output it is given under the default settings,
writes it into the clone the package was installed from, and prints the
file's path relative to it.
"""

from pathlib import Path
from pytest  import CaptureFixture, MonkeyPatch, raises

from thermur.cli                 import app
from thermur.environment.schemas import Crop, Grid, ParametricThreat, Sample
from thermur.environment.wrf     import WrfOutput
from thermur.pipeline.settings   import Settings
from thermur.repo.checkout       import Checkout


def test_a_window_missing_its_figures_writes_nothing_and_prints_nothing(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that the error a cut raises on a window missing its figures
    leaves the command with that error, no file written, and nothing
    printed.
    """

    def refused(path: Path, *, sample: Sample, settings: Settings) -> WrfOutput:
        """
        Raises the error a cut raises on a window in which no cell burns.
        """
        raise ValueError("The window holds a snapshot in which no cell burns")

    monkeypatch.setattr(Checkout, "installed", lambda: Checkout(root=tmp_path))
    monkeypatch.setattr(WrfOutput, "from_sample", refused)

    with raises(ValueError, match="no cell burns"):
        app(["cut", "sample", "wrfout_W5F7R4"], result_action="return_value")

    assert not (tmp_path / Sample().folder).exists()
    assert capsys.readouterr().out == ""


def test_another_output_than_the_sample_s_writes_nothing_and_prints_nothing(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that the command refuses an output named other than the one the
    sample is cut from, leaving no file written and nothing printed.
    """
    monkeypatch.setattr(Checkout, "installed", lambda: Checkout(root=tmp_path))

    with raises(ValueError, match="`wrfout_W5F7R3` is not `wrfout_W5F7R4`"):
        app(["cut", "sample", "wrfout_W5F7R3"], result_action="return_value")

    assert not (tmp_path / Sample().folder).exists()
    assert capsys.readouterr().out == ""


def test_the_sample_is_cut_into_the_clone_and_named_relative_to_it(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch,
    tmp_path    : Path
):
    """
    Asserts that the command cuts the pinned sample from the output it
    is given under the default settings, writes it under the root of the
    clone `Checkout.installed` resolves, and prints the written file's path
    relative to that root.
    """
    cuts: list[tuple[Path, Sample, Settings]] = []
    output = WrfOutput.from_threat(
        ParametricThreat(
            grid = Grid(
                columns   = 20,
                levels    = 12,
                rows      = 3,
                snapshots = 3
            )
        ),
        crop = Crop(floor_m=20.0, levels=5, spacing_m=20.0)
    )

    def cut(path: Path, *, sample: Sample, settings: Settings) -> WrfOutput:
        """
        Records what a cut is given and stands in the small output for the
        one it would cut.
        """
        cuts.append((path, sample, settings))
        return output

    monkeypatch.setattr(Checkout, "installed", lambda: Checkout(root=tmp_path))
    monkeypatch.setattr(WrfOutput, "from_sample", cut)
    app(["cut", "sample", "wrfout_W5F7R4"], result_action="return_value")

    assert cuts == [(Path("wrfout_W5F7R4"), Sample(), Settings())]
    assert (tmp_path / Sample().path).is_file()
    assert capsys.readouterr().out == "data/samples/wrf_sample.nc\n"
