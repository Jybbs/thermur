"""
Sets the example count on Hypothesis's built-in profiles, two hundred under
the `ci` profile Hypothesis loads on a CI runner and twenty-five under
`default`, which also drops the deadline the `ci` profile already drops, so
a heavily loaded machine fails no test a runner passes. It also defines the
fixtures every test folder shares, each described where it is defined.

The autouse `environment` fixture isolates every test from the machine
running it, and the collection hook lets a test open a network connection
only when it carries the `network` mark.
"""

from common.environment import CLEARED
from hypothesis         import settings
from pathlib            import Path
from pytest             import Config, Item, MonkeyPatch, TempPathFactory, fixture, mark

from thermur.repo.checkout import Checkout

settings.register_profile("ci", settings.get_profile("ci"), max_examples=200)
settings.register_profile(
    "default",
    settings.get_profile("default"),
    deadline     = None,
    max_examples = 25
)


@fixture
def checkout(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    tmp_path     : Path
) -> Checkout:
    """
    Copies into `tmp_path` every file the audit's checks read, meaning the
    manifest, the README, the license, `.mise/`, and `src/`, then trusts the
    copy's mise configuration and returns the `Checkout` rooted there, which
    a case edits before a check reads it.
    """
    for name in (".mise", "LICENSE", "README.md", "pyproject.toml", "src"):
        (pytestconfig.rootpath / name).copy(
            tmp_path / name,
            follow_symlinks   = False,
            preserve_metadata = True
        )

    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(tmp_path))
    return Checkout(root=tmp_path)


@fixture(autouse=True)
def environment(
    monkeypatch      : MonkeyPatch,
    pytestconfig     : Config,
    tmp_path_factory : TempPathFactory
):
    """
    Clears every shell variable `CLEARED` names, any of which would let
    the machine running the suite change a result. Some set whether a
    console prints color, `GITHUB_OUTPUT` and `GITHUB_STEP_SUMMARY` name the
    files a GitHub Actions runner collects a workflow step's outputs and a
    workflow run's summary page from, and the rest name the folders where
    a tool keeps its cache, configuration, data, and state, which the XDG
    convention defines. It then points `HOME` at an empty folder the test
    owns, so a tool that falls back to `~` finds none of the developer's
    files there.

    mise reads a configuration file out of every folder above the one
    it runs in, as well as the file at `HOME`. Moving `HOME` leaves the
    developer's own configuration above the checkout as an ordinary ancestor
    rather than as the global file, and mise stops on it as one it was never
    told to trust. The two `MISE_` variables name that file as one to skip
    and this checkout as one to trust, so a test can run mise over it.
    """
    for name in CLEARED:
        monkeypatch.delenv(name, raising=False)

    settled = Path.home() / ".config" / "mise" / "config.toml"

    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setenv("MISE_IGNORED_CONFIG_PATHS", str(settled))
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", str(pytestconfig.rootpath))


def pytest_collection_modifyitems(items: list[Item]):
    """
    Adds pytest-socket's `enable_socket` mark to every test carrying the
    `network` mark. The `--disable-socket` option in `addopts` blocks every
    test from opening a connection, and a test carrying `enable_socket` can
    reach a live service again.
    """
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(mark.enable_socket)
