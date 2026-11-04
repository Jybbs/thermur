"""
Pins the commands `thermur fetch` runs, meaning that `thermur fetch sample`
fetches the sample into the clone the package was installed from and prints
the output's path relative to it.
"""

from pathlib import Path
from pytest  import CaptureFixture, MonkeyPatch, raises

from thermur.cli                 import app
from thermur.environment.schemas import Sample
from thermur.repo.checkout       import Checkout


def test_a_failed_fetch_raises_its_own_error_and_prints_nothing(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that an error a fetch raises, such as a digest other than the
    pinned one, leaves the command with that error and nothing printed.
    """

    def refused(self: Sample, root: Path) -> Path:
        """
        Raises the error a fetch raises on an archive with another digest.
        """
        raise ValueError(f"`{self.archive}` hashes to another digest")

    monkeypatch.setattr(Sample, "fetch", refused)

    with raises(ValueError, match="hashes to another digest"):
        app(["fetch", "sample"], result_action="return_value")

    assert capsys.readouterr().out == ""


def test_the_sample_is_fetched_into_the_clone_and_named_relative_to_it(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch
):
    """
    Asserts that the command fetches the sample under the root of the clone
    `Checkout.installed` resolves and prints the extracted output's path
    relative to that root, the form the `simulation` setting takes.
    """
    roots: list[Path] = []

    def fetched(self: Sample, root: Path) -> Path:
        """
        Records the root a fetch is given and names the output it would
        extract there.
        """
        roots.append(root)
        return root / self.folder / self.member

    monkeypatch.setattr(Sample, "fetch", fetched)
    app(["fetch", "sample"], result_action="return_value")

    assert roots == [Checkout.installed().root]
    assert capsys.readouterr().out == "data/samples/wrf_sample.nc\n"
