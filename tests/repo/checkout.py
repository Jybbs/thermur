"""
Pins what `Checkout` reads from the clone, meaning the tasks mise lists
for it, the programs on its path, the commit checked out, the digest of its
`uv.lock`, and the clone the package was installed from.
"""

from common.edits import append, task
from os           import pathsep
from pathlib      import Path
from pytest       import Config, MonkeyPatch, TempPathFactory, raises
from subprocess   import CalledProcessError, check_output

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Task


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


def test_installed_roots_a_checkout_at_the_clone_the_package_came_from(
    pytestconfig: Config
):
    """
    Asserts that the installed checkout sits at the worktree `uv sync`
    installed the package from, which is the root the suite runs in.
    """
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


def test_scripts_lists_each_program_but_a_dotfile(checkout: Checkout):
    """
    Asserts that the programs are every file in the folder `_.path` names,
    relative to the root and in name order, leaving out a `.DS_Store`.
    """
    (checkout.root / ".mise/bin/.DS_Store").write_bytes(b"\x00")

    assert checkout.scripts == [Path(".mise/bin/prose"), Path(".mise/bin/pytest")]


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

    assert Task(file=path, run=(), source=path) in checkout.tasks


def test_tasks_lists_every_task_file_under_the_root(checkout: Checkout):
    """
    Asserts that each file under `.mise/tasks/` reads as a task held in that
    file and declared there, at the path mise resolves.
    """
    folder = checkout.root / ".mise/tasks"
    files  = {path for path in folder.rglob("*") if path.is_file()}

    assert set(checkout.tasks) == {
        Task(file=path, run=(), source=path) for path in files
    }


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
