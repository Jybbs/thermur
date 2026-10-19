"""
Pins what `Checkout` reads from the clone and lists for the `repo` tasks
and the pipeline:

- The tasks mise lists for the clone and the programs on its path
- The check run and the task of each pull-request row, the steps its
  workflows and composite actions run, its issue templates, and its rulesets
- The commit checked out, the digest of its `uv.lock`, and the clone the
  package was installed from
- The files git tracks there, the commands it runs from its root, and the
  scratch clone it makes of itself
"""

from common.edits import append, task
from os           import X_OK, access, pathsep
from pathlib      import Path
from pytest       import Config, MonkeyPatch, TempPathFactory, mark, param, raises
from re           import findall
from subprocess   import CalledProcessError, check_output

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Task


def test_a_clone_carries_the_checkouts_changes_and_its_programs_modes(
    clone            : Checkout,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that a clone of a checkout holds the commit checked out there
    beside the checkout's uncommitted edit, its untracked file, and its
    removal of a tracked file, while every program under `.mise/bin/` stays
    executable.
    """
    append(clone, "README.md", "\nAn uncommitted line.\n")
    append(clone, "notes.md", "An untracked note.\n")
    (clone.root / "LICENSE").unlink()

    copied = clone.clone(tmp_path_factory.mktemp("scratch") / "clone")

    assert copied.commit == clone.commit
    assert copied.read(Path("README.md")) == clone.read(Path("README.md"))
    assert copied.read(Path("notes.md")) == "An untracked note.\n"
    assert not (copied.root / "LICENSE").exists()
    assert all(access(copied.root / path, X_OK) for path in copied.scripts)


def test_commit_outside_a_clone_stops_on_the_error_git_exits_with(checkout: Checkout):
    """
    Asserts that reading the commit of a folder no git repository holds
    raises the error git exits with, rather than returning what git prints.
    """
    with raises(CalledProcessError):
        checkout.commit


def test_commit_reads_the_commit_checked_out(clone: Checkout):
    """
    Asserts that the commit is the hash git logs for the clone's `HEAD`,
    without the line break git prints after it.
    """
    assert clone.commit == check_output(
        ["git", "log", "-1", "--format=%H"],
        cwd  = clone.root,
        text = True
    ).removesuffix("\n")


def test_copy_writes_a_link_as_a_link_over_what_the_folder_holds(
    checkout         : Checkout,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that copying a link into a folder already holding a link at its
    path leaves the copied link there rather than raising.
    """
    (checkout.root / "linked.md").symlink_to("README.md")
    folder = tmp_path_factory.mktemp("copied")
    (folder / "linked.md").symlink_to("LICENSE")

    checkout.copy([Path("linked.md")], folder)

    assert (folder / "linked.md").readlink() == Path("README.md")


def test_files_lists_what_git_tracks_or_would_track_and_still_finds(clone: Checkout):
    """
    Asserts that the files a checkout lists take in an untracked file and
    one whose name git quotes, and leave out one its `.gitignore` covers and
    a tracked one removed from the disk.
    """
    append(clone, "notes.md", "An untracked note.\n")
    append(clone, "café.md", "A note whose name git quotes.\n")
    append(clone, ".env", "IGNORED=1\n")
    (clone.root / "LICENSE").unlink()

    files = clone.files

    assert Path("notes.md") in files
    assert Path("café.md") in files
    assert Path(".env") not in files
    assert Path("LICENSE") not in files


@mark.parametrize(
    ("folder", "member", "names"),
    [
        param(
            ".github/ISSUE_TEMPLATE",
            "templates",
            ["bug.md", "spec.md"],
            id = "templates"
        ),
        param(".mise/bin", "scripts", ["prose", "pytest"], id="scripts")
    ]
)
def test_a_listing_leaves_out_a_dotfile(
    checkout : Checkout,
    folder   : str,
    member   : str,
    names    : list[str]
):
    """
    Asserts that the issue templates and the programs are every file in
    their folder, relative to the root and in name order, leaving out a
    `.DS_Store`, which holds no UTF-8 text.
    """
    (checkout.root / folder / ".DS_Store").write_bytes(b"\x00\xff")

    assert list(getattr(checkout, member)) == [Path(folder, name) for name in names]


def test_installed_roots_a_checkout_at_the_clone_the_package_came_from(
    monkeypatch  : MonkeyPatch,
    pytestconfig : Config,
    tmp_path     : Path
):
    """
    Asserts that the installed checkout sits at the worktree `uv sync`
    installed the package from, whatever the working directory.
    """
    monkeypatch.chdir(tmp_path)

    assert Checkout.installed().root == pytestconfig.rootpath.resolve()


def test_lock_digest_is_the_sha256_digest_of_uv_lock(checkout: Checkout):
    """
    Asserts that the lock digest is the SHA-256 digest of `uv.lock`, pinned
    on an empty file against the digest of empty input.
    """
    (checkout.root / "uv.lock").write_bytes(b"")

    assert checkout.lock_digest == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_rows_lists_the_checks_of_pull_request_workflows_alone(checkout: Checkout):
    """
    Asserts that the checks are the matrix rows of each workflow running on
    a pull request, each as the name of its check run and its task, and the
    rows are those tasks, both in the order the rows are declared, leaving
    out a row of a workflow running only on a push and a row naming no task.
    """
    folder = checkout.root / ".github/workflows"
    for path in folder.iterdir():
        path.unlink()

    matrix = (
        "jobs:\n  a:\n    name: ${{ matrix.name }}\n    runs-on: x\n"
        "    strategy:\n      matrix:\n        include:\n"
    )
    (folder / "pull.yml").write_text(
        f"name: pull\non:\n  pull_request:\n{matrix}"
        "          - { name: B, task: b:two }\n          - { name: A, task: a:one }\n"
        "          - { name: K, tools: x }\n"
    )
    (folder / "push.yml").write_text(
        f"name: push\non:\n  push:\n{matrix}          - {{ name: C, task: c:three }}\n"
    )

    assert checkout.checks == [("B", "b:two"), ("A", "a:one")]
    assert checkout.rows == ["b:two", "a:one"]


def test_rulesets_reads_each_ruleset_by_its_path(checkout: Checkout):
    """
    Asserts that the rulesets are the JSON files under `.github/rulesets/`,
    each by its path from the root in path order and validated under the
    name it carries on GitHub.
    """
    assert [(path, ruleset.name) for path, ruleset in checkout.rulesets.items()] == [
        (Path(".github/rulesets/main.json"), "main branch protection"),
        (Path(".github/rulesets/tags.json"), "release tag protection")
    ]


def test_run_leaves_out_the_environment_a_task_runs_in(
    checkout    : Checkout,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a command the checkout runs sees no `VIRTUAL_ENV`, so the
    `.venv` a task's `uv run` names for the clone it runs from never stands
    in for the root's own.
    """
    monkeypatch.setenv("VIRTUAL_ENV", "/elsewhere/.venv")

    assert checkout.run(
        "sh",
        "-c",
        'printf %s "${VIRTUAL_ENV-unset}"',
        capture = True
    ).stdout == "unset"


def test_run_runs_in_its_folder_and_raises_only_where_checked(checkout: Checkout):
    """
    Asserts that a command the checkout runs starts in the folder named
    under the root, and that one exiting nonzero raises unless the call
    leaves its status unchecked.
    """
    assert checkout.run("pwd", capture=True, folder=Path("docs")).stdout.strip() == (
        str(checkout.root / "docs")
    )
    assert checkout.run("false", check=False).returncode == 1
    with raises(CalledProcessError):
        checkout.run("false")


def test_steps_lists_the_actions_ahead_of_the_workflows(checkout: Checkout):
    """
    Asserts that every step reads beside its file's path from the root, the
    composite action's steps ahead of the workflows' in path order.
    """
    paths = [path for path, _ in checkout.steps]

    assert paths == sorted(paths)
    assert set(paths) == {*checkout.actions, *checkout.workflows}


def test_tasks_carries_the_lines_a_toml_task_runs(checkout: Checkout):
    """
    Asserts that a task `.mise/config.toml` declares in TOML reads as one
    holding no file, declared in that configuration, and running its lines.
    """
    append(
        checkout,
        ".mise/config.toml",
        '\n[tasks."py:sweep"]\nrun = ["prose check src", "pytest"]\n'
    )

    assert Task(
        file   = None,
        name   = "py:sweep",
        run    = ("prose check src", "pytest"),
        source = checkout.root / ".mise/config.toml"
    ) in checkout.tasks


def test_tasks_leaves_out_a_task_declared_above_the_root(
    checkout         : Checkout,
    monkeypatch      : MonkeyPatch,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that a task a configuration above the checkout declares, which
    mise lists for the checkout as well, is left out of the checkout's
    tasks, which match the tasks the original copy declares.
    """
    above = tmp_path_factory.mktemp("above")
    (above / "mise.toml").write_text('[tasks."above:task"]\nrun = "true"\n')
    clone = checkout.root.copy(
        above / "clone",
        follow_symlinks   = False,
        preserve_metadata = True
    )
    monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", f"{above}{pathsep}{checkout.root}")
    cloned = Checkout(root=clone)

    assert above / "mise.toml" in {task.source for task in cloned.listed}
    assert {task.source.relative_to(clone) for task in cloned.tasks} == {
        task.source.relative_to(checkout.root) for task in checkout.tasks
    }


def test_tasks_lists_a_hidden_task(checkout: Checkout):
    """
    Asserts that a task its `#MISE hide` line keeps out of a plain `mise
    tasks ls` still reads as one of the checkout's tasks, so every check
    holds it as it holds a listed one.
    """
    path = checkout.root / task(checkout, "py/hidden", "#!/bin/sh\n#MISE hide = true\n")

    assert Task(
        file   = path,
        name   = "py:hidden",
        run    = (),
        source = path
    ) in checkout.tasks


def test_tasks_lists_every_task_file_under_the_root(checkout: Checkout):
    """
    Asserts that each executable file under `.mise/tasks/` reads as a task
    held in that file and declared there, at the path mise resolves, named
    for its folder and its stem, beside the alias its `MISE alias` line
    declares.
    """
    folder = checkout.root / ".mise/tasks"
    files  = {path for path in folder.rglob("*") if path.is_file() and access(
        path,
        X_OK
    )}

    assert set(checkout.tasks) == {
        Task(
            aliases = tuple(
                findall(r'(?m)^# ?MISE alias *= "([^"]+)"', path.read_text())
            ),
            file   = path,
            name   = ":".join(path.relative_to(folder).with_suffix("").parts),
            run    = (),
            source = path
        )
        for path in files
    }


@mark.parametrize(
    ("trusted", "reads"),
    [
        param("/elsewhere", "/elsewhere{sep}{root}", id="beside"),
        param(None, "{root}", id="alone")
    ]
)
def test_run_trusts_the_root_beside_each_configuration_already_trusted(
    checkout    : Checkout,
    monkeypatch : MonkeyPatch,
    reads       : str,
    trusted     : str | None
):
    """
    Asserts that a command the checkout runs sees mise trusting the root's
    configuration after each one the environment already trusts.
    """
    monkeypatch.delenv("MISE_TRUSTED_CONFIG_PATHS")
    if trusted:
        monkeypatch.setenv("MISE_TRUSTED_CONFIG_PATHS", trusted)

    printed = checkout.run(
        "sh",
        "-c",
        'printf %s "$MISE_TRUSTED_CONFIG_PATHS"',
        capture = True
    )

    assert printed.stdout == reads.format(root=checkout.root, sep=pathsep)


def test_the_root_resolves_through_a_symlink(
    checkout         : Checkout,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that a checkout rooted at a symlink holds the folder the link
    resolves to, the form in which mise names each task's file.
    """
    link = tmp_path_factory.mktemp("links") / "clone"
    link.symlink_to(checkout.root)

    assert Checkout(root=link).root == checkout.root


def test_unwrap_blanks_a_comment_and_keeps_the_blank_line_above_it():
    """
    Asserts that reading the lines a shell runs blanks an indented comment
    and keeps the blank line above it, each line keeping its place.
    """
    assert Checkout.unwrap("a\n\n  # c\nb\n") == "a\n\n\nb\n"
