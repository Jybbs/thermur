"""
Pins what `lock:lift` reads from each package manager, each answering
through a stand-in program over a copy of the checkout, and what the lift
then prints, writes, and tries:

- The pins `Tools`, `Locked`, `Scripts`, `Builds`, `Site`, and `Actions`
  report lagging, each beside its newest release's date and the advisories
  the lift clears, and the edits each writes into the copy
- The table the lift prints, the holds naming no row, and the pins it
  lifts short of its holds
- The commands the write runs, and the trial's verdict in a scratch clone
  beside the major lift whose hold lets the rest pass
"""

from collections.abc  import Callable
from common.edits     import edit
from common.lags      import BREAKING, uv_lag
from common.stand_ins import StandIn
from contextlib       import nullcontext
from datetime         import UTC, datetime
from json             import dumps, loads
from pathlib          import Path
from pytest           import MonkeyPatch, fixture, mark, param, raises
from shlex            import quote
from shutil           import which
from subprocess       import CalledProcessError
from tomllib          import loads as from_toml

from thermur.repo.checkout import Checkout
from thermur.repo.checks   import ParityCheck
from thermur.repo.lift     import (
    Actions,
    Builds,
    Lift,
    Locked,
    Manager,
    Scripts,
    Site,
    Tools
)
from thermur.repo.schemas  import Config, Lag, Manifest

ADVISED = {"satori": [{"url": "https://github.com/advisories/GHSA-s"}]}
BRIEF   = ".mise/tasks/gha/brief.py"
CONFIG  = Path(".mise/config.toml")
NOW     = datetime(2026, 10, 16, tzinfo=UTC)


@fixture
def answer(stand_in: Callable[[str, str], StandIn]) -> Callable[..., StandIn]:
    """
    Writes stand-in programs that answer by their arguments.

    Returns:
        A writer taking the program's name and, by each `case` pattern its
        arguments may match, the `sh` lines it runs for the first one they
        match, running nothing where none does.
    """
    def write(program: str, *, answers: dict[str, str]) -> StandIn:
        """
        Writes the stand-in `program` running the lines `answers` gives
        beside the first pattern its arguments match.
        """
        return stand_in(
            program,
            'case "$*" in\n'
            + "".join(
                f"  {pattern}) {lines} ;;\n" for pattern, lines in answers.items()
            )
            + "esac"
        )

    return write


@fixture
def floor(checkout: Checkout) -> Lag:
    """
    Builds the lag lifting the floor `min_version` sets in the copy to the
    last release of the year it names, which no major lift reaches.
    """
    return Lag.declared(
        {CONFIG: r'^min_version\s*=\s*"'},
        current = checkout.config.floor,
        newest  = f"{checkout.config.floor.partition('.')[0]}.99.0",
        pin     = "mise"
    )


def package(name: str, *, uploaded: str, version: str) -> dict:
    """
    Writes one package as `uv tree --format json` lists it under
    `resolution`, carrying one wheel uploaded at `uploaded`.
    """
    return {
        "kind"    : "package",
        "name"    : name,
        "version" : version,
        "wheels"  : [{"upload_time": uploaded}]
    }


def printed(value: object) -> str:
    """
    Writes the `sh` line printing `value`, as JSON unless it is a string.
    """
    return f"printf '%s\\n' {quote(value if isinstance(value, str) else dumps(value))}"


def test_a_manager_reading_nothing_of_its_own_reports_no_lag(checkout: Checkout):
    """
    Asserts that the base manager, running no report command of its own,
    reports no pin lagging, the default each manager builds on.
    """
    assert Manager(checkout=checkout).lags == []


def test_a_python_lift_within_its_minor_leaves_its_floor(
    answer   : Callable[..., StandIn],
    checkout : Checkout
):
    """
    Asserts that a `python` lift keeping its minor writes the pin alone,
    leaving the `requires-python` floor and every restatement of it as they
    stood.
    """
    python = from_toml(checkout.read(CONFIG))["tools"]["python"]
    patch  = f"{checkout.config.python}.99"
    answer(
        "mise",
        answers = {
            '"outdated --bump --json --local"': printed(
                value = {"python": {"latest": patch, "requested": python}}
            ),
            '"outdated "*'  : printed({}),
            '"ls-remote "*' : printed([])
        }
    )
    manifest = checkout.read(Path("pyproject.toml"))

    lags = written(Tools(checkout=checkout).lags, checkout.root)

    assert [lag.files for lag in lags] == [[CONFIG]]
    assert from_toml(checkout.read(CONFIG))["tools"]["python"] == patch
    assert checkout.read(Path("pyproject.toml")) == manifest


def test_an_action_under_a_subpath_reads_the_releases_of_its_repository(
    answer   : Callable[..., StandIn],
    checkout : Checkout
):
    """
    Asserts that an action used under a subpath of its repository reads the
    latest release and the tags of the repository its owner and name give,
    reports the lag under the full action, and lifts its pin in the file
    naming it.
    """
    lifted = "b" * 40
    (checkout.root / ".github/workflows/scan.yml").write_text(
        "name: s\non:\n  push:\njobs:\n  a:\n    runs-on: x\n    steps:\n"
        f"      - uses: github/codeql-action/init@{'a' * 40}\n"
    )
    pins = dict(step.pin for _, step in checkout.steps if step.pin)
    gh   = answer(
        "gh",
        answers = {
            "*codeql-action": printed(
                value = {"publishedAt": NOW.isoformat(), "tagName": "v4"}
            ),
            "*": printed({"publishedAt": NOW.isoformat(), "tagName": "v1"})
        }
    )
    answer(
        "git",
        answers = {
            "*/github/codeql-action": printed(f"{lifted}\trefs/tags/v4"),
            **{
                f"*/{action}": printed(f"{commit}\trefs/tags/v1")
                for action, commit in pins.items()
            }
        }
    )

    lags = written(Actions(checkout=checkout).lags, checkout.root)

    assert [(lag.pin, lag.newest, lag.files) for lag in lags] == [
        ("github/codeql-action/init", "v4", [Path(".github/workflows/scan.yml")])
    ]
    assert "--repo github/codeql-action" in " ".join(gh.commands)
    assert f"github/codeql-action/init@{lifted}" in checkout.read(
        Path(".github/workflows/scan.yml")
    )


def test_builds_reads_each_build_pin_against_unpinned_hatchling(
    answer      : Callable[..., StandIn],
    checkout    : Checkout,
    monkeypatch : MonkeyPatch,
    pins        : dict[str, str]
):
    """
    Asserts that `Builds` resolves the build requirement unpinned and
    reports each build pin the resolution moves past, leaving out one it
    resolves at its pin and one it does not resolve, and that writing its
    edit lifts the constraint in `pyproject.toml`.
    """
    resolved = checkout.root.parent / "resolved.py"
    tree     = {
        "resolution": {
            "h" : package(
                name     = "hatchling",
                uploaded = "2026-09-20T00:00:00Z",
                version  = pins["hatchling"]
            ),
            "p" : package("pluggy", uploaded="2026-10-15T00:00:00Z", version="9.9.9")
        }
    }
    monkeypatch.setenv("RESOLVED", str(resolved))
    answer(
        "uv",
        answers = {
            '"tree --format"*': 'for script; do :; done; cat "$script" > "$RESOLVED"; '
            + printed(tree)
        }
    )

    lags = written(Builds(checkout=checkout).lags, checkout.root)

    assert [(lag.pin, lag.current, lag.newest, lag.published) for lag in lags] == [
        ("pluggy", pins["pluggy"], "9.9.9", datetime(2026, 10, 15, tzinfo=UTC))
    ]
    assert '# dependencies    = ["hatchling"]' in resolved.read_text()
    assert f'# requires-python = ">={checkout.manifest.floor}"' in resolved.read_text()
    assert "pluggy==9.9.9" in map(str, Manifest.read(checkout.root).constraints)


def test_locked_reads_each_package_the_upgrade_moves(
    answer   : Callable[..., StandIn],
    checkout : Checkout
):
    """
    Asserts that `Locked` reports each package the dry run of `uv lock
    --upgrade` updates, leaving out one it adds, as a package the lift
    upgrades, beside the earliest upload of its newest release and each
    advisory a release up to the newest fixes.
    """
    dry_run = (
        "Resolved 3 packages in 1ms\n"
        "Update alpha v1.0.0 -> v1.1.0\n"
        "Add gamma v1.0.0\n"
        "Update beta-pkg v0.9.0 -> v2.0.0"
    )
    alpha = {"name": "alpha", "version": "1.0.0"}
    audit = {
        "vulnerabilities": [
            {"dependency": alpha, "id": "GHSA-fixed", "fix_versions": ["1.0.5"]},
            {"dependency": alpha, "id": "GHSA-later", "fix_versions": ["1.2.0"]}
        ]
    }
    tree = {
        "resolution": {
            "a" : package("alpha", uploaded="2026-10-07T00:00:00Z", version="1.1.0"),
            "b" : package("beta-pkg", uploaded="2026-10-09T00:00:00Z", version="2.0.0")
        }
    }
    answer(
        "uv",
        answers = {
            '"lock --dry-run"*' : f"{printed(dry_run)} >&2",
            '"audit "*'         : printed(audit),
            '"tree --format"*'  : printed(tree)
        }
    )

    assert Locked(checkout=checkout).lags == [
        Lag(
            advisories = ("GHSA-fixed",),
            current    = "1.0.0",
            newest     = "1.1.0",
            pin        = "alpha",
            published  = datetime(2026, 10, 7, tzinfo=UTC),
            upgraded   = True
        ),
        Lag(
            current   = "0.9.0",
            newest    = "2.0.0",
            pin       = "beta-pkg",
            published = datetime(2026, 10, 9, tzinfo=UTC),
            upgraded  = True
        )
    ]


def test_the_lift_gathers_each_managers_lags_in_the_order_they_are_declared(
    checkout    : Checkout,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a lift given no lags gathers those of every manager,
    each found as a subclass of `Manager`, in the order the managers are
    declared.
    """
    for manager in Manager.__subclasses__():
        monkeypatch.setattr(
            manager,
            "lags",
            property(
                lambda self: [Lag(current="1", newest="2", pin=type(self).__name__)]
            )
        )

    assert [lag.pin for lag in Lift(checkout=checkout).lags] == [
        "Actions", "Builds", "Locked", "Scripts", "Site", "Tools"
    ]


@mark.parametrize(
    ("latest", "listed"),
    [
        param("9.0.0", True, id="behind"),
        param(None, True, id="newest"),
        param("9.0.0", False, id="unlisted")
    ]
)
def test_scripts_reads_each_exact_pin_a_task_script_declares(
    answer   : Callable[..., StandIn],
    checkout : Checkout,
    latest   : str | None,
    listed   : bool,
    pins     : dict[str, str]
):
    """
    Asserts that `Scripts` reports the exact pin a task script declares
    where `uv tree --outdated` names a newer release, beside that release's
    upload and each advisory on the script it clears, and that writing its
    edit lifts the pin in the script, whereas a pin at its newest release,
    and one whose package the tree lists for no platform it resolves, make
    no row.
    """
    pinned = pins["minijinja"]
    audit  = {
        "vulnerabilities": [
            {
                "id"           : "GHSA-m",
                "dependency"   : {"name": "minijinja", "version": pinned},
                "fix_versions" : ["9.0.0"]
            }
        ]
    }
    outdated = {
        "resolution": {
            "m": {
                "kind"           : "package",
                "latest_version" : latest,
                "name"           : "minijinja",
                "version"        : pinned
            }
        }
        if listed
        else {}
    }
    resolved = {
        "resolution": {
            "m": package("minijinja", uploaded="2026-10-15T00:00:00Z", version="9.0.0")
        }
    }
    answer(
        "uv",
        answers = {
            '"audit "*"--script "*'  : printed(audit),
            '"tree "*"--outdated"*'  : printed(outdated),
            '"tree "*"--no-locked"*' : printed(resolved)
        }
    )

    lags = written(Scripts(checkout=checkout).lags, checkout.root)

    assert [(lag.pin, lag.current, lag.advisories, lag.published) for lag in lags] == (
        [("minijinja", pinned, ("GHSA-m",), datetime(2026, 10, 15, tzinfo=UTC))]
        if latest and listed
        else []
    )
    assert ('"minijinja==9.0.0"' in checkout.read(
        Path(BRIEF)
    )) is bool(latest and listed)


def test_the_lift_lifts_each_pin_no_hold_names_and_lists_each_hold_naming_none(
    checkout: Checkout
):
    """
    Asserts that the lift keeps each lagging pin neither its holds nor the
    pins a trial holds name, a hold naming every row carrying the pin, and
    lists in order each hold naming no row.
    """
    lags = [
        Lag(
            current  = "1",
            newest   = "2",
            pin      = "minijinja",
            upgraded = True
        ),
        Lag(current="1", newest="2", pin="minijinja"),
        Lag(current="1", newest="2", pin="satori"),
        Lag(current="1", newest="2", pin="vue")
    ]
    lift = Lift(checkout=checkout, holds=("zeta", "minijinja", "alpha"), lags=lags)

    assert [lag.pin for lag in lift.lifted()] == ["satori", "vue"]
    assert [lag.pin for lag in lift.lifted({"vue"})] == ["satori"]
    assert lift.unmatched == ["alpha", "zeta"]


@mark.parametrize(
    ("audited", "after", "cleared"),
    [
        param({}, {}, (), id="no-advisory"),
        param(ADVISED, {}, ("GHSA-s",), id="cleared"),
        param(ADVISED, ADVISED, (), id="persisting")
    ]
)
def test_site_reads_each_package_bun_outdated_lists(
    after    : dict,
    answer   : Callable[..., StandIn],
    audited  : dict,
    checkout : Checkout,
    cleared  : tuple[str, ...]
):
    """
    Asserts that `Site` reports each package `bun outdated` lists, with or
    without its `(dev)` mark, beside the date `bun info` gives its newest
    release, and clears an advisory `bun audit` reports on one only where
    the audit of a copy of `docs/` with the lift written reports it no
    longer, a copy it resolves only where an advisory names a row, so an
    advisory the lift leaves standing clears nothing.
    """
    pins  = loads(checkout.read(Path("docs/package.json")))["devDependencies"]
    table = (
        "| Package | Current | Update | Latest |\n"
        "|---------|---------|--------|--------|\n"
        f"| satori (dev) | {pins['satori']} | {pins['satori']} | 0.99.0 |\n"
        f"| vue | {pins['vue']} | {pins['vue']} | 9.9.9 |"
    )
    times  = {"0.99.0": "2026-10-16T08:03:00.061Z", "9.9.9": "2026-10-13T00:00:00Z"}
    lifted = 'grep -q \'"satori"[^,]*"0.99.0"\' package.json'
    bun    = answer(
        "bun",
        answers = {
            "outdated"       : printed(table),
            '"info "*'       : printed(times),
            '"audit --json"' : f'case "$PWD" in */thermur-*) {lifted} && '
            f"{printed(after)} || {printed(audited)} ;; *) {printed(audited)} ;; esac"
        }
    )

    lags = Site(checkout=checkout).lags

    assert [(lag.pin, lag.newest, lag.published, lag.advisories) for lag in lags] == [
        (
            "satori", "0.99.0",
            datetime(2026, 10, 16, 8, 3, 0, 61000, tzinfo=UTC),
            cleared
        ),
        ("vue", "9.9.9", datetime(2026, 10, 13, tzinfo=UTC), ())
    ]
    assert ("bun install --lockfile-only" in bun.commands) is bool(audited)


@mark.parametrize(
    ("tags", "current"),
    [
        param(
            ["{pin}\trefs/tags/v7", "{pin}\trefs/tags/v7.0.1"],
            "v7.0.1",
            id = "tagged"
        ),
        param([], None, id="untagged")
    ]
)
def test_actions_reads_each_action_its_latest_release_moves(
    answer   : Callable[..., StandIn],
    checkout : Checkout,
    current  : str | None,
    tags     : list[str]
):
    """
    Asserts that `Actions` reports an action whose latest release tags
    a commit other than its pin, naming the current release by the most
    specific tag naming the pinned commit or by the commit's opening, and
    reading an annotated tag as the commit it annotates, while an action
    whose latest release tags its pin makes no row. Writing its edits lifts
    the pin in every file naming it.
    """
    pins    = dict(step.pin for _, step in checkout.steps if step.pin)
    pinned  = pins.pop("actions/upload-artifact")
    lifted  = "f" * 40
    listing = [
        *(tag.format(pin=pinned) for tag in tags),
        f"{'e' * 40}\trefs/tags/v9.9.9",
        f"{lifted}\trefs/tags/v9.9.9^{{}}"
    ]
    answer(
        "gh",
        answers = {
            "*upload-artifact": printed(
                value = {"publishedAt": NOW.isoformat(), "tagName": "v9.9.9"}
            ),
            "*": printed({"publishedAt": NOW.isoformat(), "tagName": "v1"})
        }
    )
    answer(
        "git",
        answers = {
            "*/actions/upload-artifact": printed("\n".join(listing)),
            **{
                f"*/{action}": printed(f"{commit}\trefs/tags/v1")
                for action, commit in pins.items()
            }
        }
    )

    lags = written(Actions(checkout=checkout).lags, checkout.root)

    assert [(lag.pin, lag.current, lag.newest, lag.files) for lag in lags] == [
        (
            "actions/upload-artifact",
            current or pinned[:7],
            "v9.9.9",
            [
                Path(".github/workflows/ci.yml"),
                Path(".github/workflows/pages.yml"),
                Path(".github/workflows/site.yml")
            ]
        )
    ]
    assert {
        commit
        for _, step in Checkout(root=checkout.root).steps
        if step.pin
        for action, commit in [step.pin]
        if action == "actions/upload-artifact"
    } == {lifted}


def test_the_table_lists_each_lag_or_says_none_lags(checkout: Checkout):
    """
    Asserts that the lift prints one Markdown table holding a row for each
    lagging pin under a bolded header, and a sentence in its place where no
    pin lags.
    """
    uv = uv_lag(checkout).model_copy(
        update = {"published": datetime(2026, 10, 14, tzinfo=UTC)}
    )
    lags = [
        uv,
        Lag(
            current  = "2.24.0",
            newest   = "2.25.0",
            pin      = "minijinja",
            upgraded = True
        )
    ]

    assert Lift(checkout=checkout, lags=lags).table(NOW).splitlines() == [
        "| **Pin** | **Declared In** | **Current** | **Newest** | **Major** "
        "| **Age in Days** | **Clears** |",
        "|---|---|---|---|---|---|---|",
        f"| `uv` | `.mise/config.toml` | {uv.current} | {uv.newest} | yes | 2 |  |",
        "| `minijinja` | `uv.lock` | 2.24.0 | 2.25.0 | no |  |  |"
    ]
    assert Lift(checkout=checkout, lags=[]).table(NOW) == (
        "Every pin stands at its newest release."
    )


@mark.parametrize(
    ("upgraded", "commands"),
    [
        param(
            True,
            [
                "mise lock", "mise install",
                "mise x -- uv lock --no-locked --upgrade-package alpha",
                "mise run lock:sync"
            ],
            id = "upgraded"
        ),
        param(False, ["mise lock", "mise install", "mise run lock:sync"], id="edited")
    ]
)
def test_the_write_lifts_each_pin_then_locks_and_relocks(
    checkout : Checkout,
    commands : list[str],
    mise     : StandIn,
    upgraded : bool
):
    """
    Asserts that the write rewrites each pin the lift lifts, leaving a
    held one, then locks and installs the tools, upgrades each package of
    `uv.lock` the lift moves short of those held, and relocks the rest.
    """
    packages = [
        Lag(
            current  = "1.0",
            newest   = "1.1",
            pin      = pin,
            upgraded = True
        )
        for pin in ("alpha", "beta")
        if upgraded
    ]

    Lift(checkout=checkout, holds=("beta",), lags=[uv_lag(checkout), *packages]).write(
        checkout
    )

    assert Config.read(checkout.root).uv == uv_lag(checkout).newest
    assert mise.commands == commands


@mark.parametrize("mise", [param(BREAKING, id="failing-on-uv")], indirect=True)
def test_a_trial_passes_where_mise_ci_passes_and_isolates_the_major_lift(
    clone : Checkout,
    floor : Lag,
    mise  : StandIn
):
    """
    Asserts that a trial whose lifted clone fails `mise ci` fails, that
    one holding back the major lift breaking it passes, and that the lift
    isolates that lift, rebuilding `.venv` before each run of `mise ci` and
    leaving the checkout as it was.
    """
    lift = Lift(checkout=clone, lags=[floor, uv_lag(clone)])

    assert (lift.passes(), lift.passes({"uv"}), lift.isolate()) == (False, True, "uv")
    assert Config.read(clone.root) == clone.config
    assert mise.commands.count("mise x -- uv sync") == mise.commands.count(
        "mise run ci"
    ) == 3


@mark.parametrize(
    "mise",
    [param('[ "$1" = lock ] && exit 1\nexit 0', id="failing-lock")],
    indirect = True
)
def test_a_trial_whose_write_fails_fails_before_mise_ci_and_isolates_nothing(
    clone : Checkout,
    mise  : StandIn
):
    """
    Asserts that a trial whose write stops on a failing command fails
    without running `mise ci`, and that holding back its one major lift
    passes no trial either, so the lift isolates none.
    """
    lift = Lift(checkout=clone, lags=[uv_lag(clone)])

    assert (lift.passes(), lift.isolate()) == (False, None)
    assert "mise run ci" not in mise.commands


def test_tools_lifts_a_tool_its_configuration_pins_through_a_table(
    answer   : Callable[..., StandIn],
    checkout : Checkout
):
    """
    Asserts that a tool `.mise/config.toml` pins through an inline table,
    the form an `os` restriction takes, is lifted at the `version` the table
    names, leaving the rest of the table as it stood.
    """
    zizmor = from_toml(checkout.read(CONFIG))["tools"]["zizmor"]
    edit(
        checkout,
        str(CONFIG),
        new = f'zizmor = {{ os = ["linux"], version = "{zizmor}" }}',
        old = f'zizmor                    = "{zizmor}"'
    )
    answer(
        "mise",
        answers = {
            '"outdated --bump --json --local"': printed(
                value = {"zizmor": {"latest": "9.9.9", "requested": zizmor}}
            ),
            '"outdated "*'  : printed({}),
            '"ls-remote "*' : printed([])
        }
    )

    written(Tools(checkout=checkout).lags, checkout.root)

    assert from_toml(checkout.read(CONFIG))["tools"]["zizmor"] == {
        "version" : "9.9.9",
        "os"      : ["linux"]
    }


def test_tools_reads_each_tool_mise_reports_newer_and_the_floor(
    answer   : Callable[..., StandIn],
    checkout : Checkout
):
    """
    Asserts that `Tools` reports each tool `mise outdated --bump` names
    beside the date `mise ls-remote` gives its newest release, or none,
    then the floor under the name `mise`, and that writing its edits lifts
    each pin, one under a quoted key and the `required-version` restating
    the `uv` pin among them. A `python` lift to another minor moves the
    `requires-python` floor, each restatement `repo:audit` holds to it, and
    each task script's own floor with it, leaving the audit clean.
    """
    config = checkout.config
    pinned = from_toml(checkout.read(CONFIG))["tools"]
    lychee = "aqua:lycheeverse/lychee"
    tools  = {
        "python" : {"latest": "3.99.0", "requested": pinned["python"]},
        "uv"     : {"latest": "9.0.0", "requested": config.uv},
        lychee   : {"latest": "9.9.9", "requested": pinned[lychee]}
    }
    uv = [
        {"created_at": "2026-01-01T00:00:00Z", "version": "9.0.0-rc.1"},
        {"created_at": "2026-10-14T05:15:30.0Z", "version": "9.0.0"}
    ]
    released = [{"created_at": "2026-10-15T10:13:36Z", "version": "2099.1.0"}]
    floored  = {"aqua:jdx/mise": {"latest": "2099.1.0", "requested": config.floor}}
    answer(
        "mise",
        answers = {
            '"tasks "*': f'exec {quote(which("mise"))} "$@"',
            '"outdated --bump --json --local"': printed(tools),
            f'"outdated --bump --json aqua:jdx/mise@{config.floor}"': printed(floored),
            f'"ls-remote --json {lychee}@"*'     : printed([{"version": "9.9.9"}]),
            '"ls-remote --json python@"*'        : printed([{"version": "3.99.0"}]),
            '"ls-remote --json uv@"*'            : printed(uv),
            '"ls-remote --json aqua:jdx/mise@"*' : printed(released)
        }
    )

    lags = written(Tools(checkout=checkout).lags, checkout.root)

    assert [(lag.pin, lag.current, lag.newest, lag.published) for lag in lags] == [
        ("python", pinned["python"], "3.99.0", None),
        ("uv", config.uv, "9.0.0", datetime(2026, 10, 14, 5, 15, 30, tzinfo=UTC)),
        (lychee, pinned[lychee], "9.9.9", None),
        (
            "mise", config.floor, "2099.1.0",
            datetime(2026, 10, 15, 10, 13, 36, tzinfo=UTC)
        )
    ]
    assert Config.read(checkout.root).model_dump(include={"floor", "python", "uv"}) == {
        "floor"  : "2099.1.0",
        "python" : "3.99",
        "uv"     : "9.0.0"
    }
    assert Manifest.read(checkout.root).model_dump(
        include = {"floor", "required_version", "target_version"}
    ) == {"floor": "3.99", "required_version": "==9.0.0", "target_version": "3.99"}
    assert from_toml(checkout.read(CONFIG))["tools"][lychee] == "9.9.9"
    assert '# requires-python = ">=3.99"' in checkout.read(Path(BRIEF))
    assert ParityCheck(checkout=Checkout(root=checkout.root)).findings == []


@mark.parametrize(
    ("reported", "status", "raised"),
    [
        param(True, 1, False, id="advisory"),
        param(True, 2, True, id="unreachable"),
        param(False, 1, True, id="stale")
    ]
)
def test_the_audit_reads_what_uv_reports_on_an_advisory_and_raises_past_it(
    answer   : Callable[..., StandIn],
    checkout : Checkout,
    raised   : bool,
    reported : bool,
    status   : int
):
    """
    Asserts that a manager reads the advisories `uv audit` prints where
    it exits `1` on finding one, and raises where it exits with any other
    failing status or prints no report, as it does where OSV cannot be
    reached or the lockfile is stale.
    """
    audit = {
        "vulnerabilities": [
            {
                "id"           : "GHSA-a",
                "dependency"   : {"name": "alpha", "version": "1.0.0"},
                "fix_versions" : ["1.0.5"]
            }
        ]
    }
    answer(
        "uv",
        answers = {
            '"audit "*': f"{printed(audit) if reported else 'true'}; exit {status}"
        }
    )

    with raises(CalledProcessError) if raised else nullcontext():
        assert Manager(checkout=checkout).audit().clears("1.0.0", "alpha", "1.1.0") == (
            "GHSA-a",
        )


@mark.parametrize(
    ("answered", "raised"),
    [
        param("echo 'release not found' >&2; exit 1", False, id="unreleased"),
        param("echo 'HTTP 401: Bad credentials' >&2; exit 1", True, id="refused")
    ]
)
def test_an_action_whose_repository_publishes_no_release_makes_no_row(
    answer   : Callable[..., StandIn],
    answered : str,
    checkout : Checkout,
    raised   : bool
):
    """
    Asserts that an action whose repository `gh` finds no release for makes
    no row without reading its tags, which a stand-in `git` failing every
    call stops, whereas any other failure of `gh` raises rather than leaving
    the action out of the table unreported.
    """
    answer("gh", answers={"*": answered})
    answer("git", answers={"*": "exit 1"})

    with raises(CalledProcessError) if raised else nullcontext():
        assert Actions(checkout=checkout).lags == []


def written(lags: list[Lag], root: Path) -> list[Lag]:
    """
    Writes every edit of `lags` into the copy at `root`.

    Returns:
        The lags as given.
    """
    for lag in lags:
        lag.write(root)

    return lags
