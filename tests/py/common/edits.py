"""
Holds the edits a case makes to its copy of the checkout before a check
reads it, meaning text added to the end of a file, one string replaced in a
file, the holds `pyproject.toml` records, a task file written fresh, a git
command run under a fixed author, and a repository committing every file it
holds to `main`, beside the writer of an executable file each of them and
the stand-ins share.
"""

from json    import dumps
from pathlib import Path
from re      import sub

from thermur.repo.checkout import Checkout


def append(checkout: Checkout, path: str, text: str):
    """
    Adds `text` to the end of the file at `path` under the copy.
    """
    with (checkout.root / path).open("a", encoding="utf-8") as file:
        file.write(text)


def committed(checkout: Checkout):
    """
    Starts a git repository on `main` at the root of `checkout` and commits
    every file under it there.
    """
    for command in (
        ["init", "--initial-branch=main", "--quiet"],
        ["add", "--all"],
        ["commit", "--message=Start", "--quiet"]
    ):
        git(checkout, *command)


def edit(checkout: Checkout, path: str, *, new: str, old: str):
    """
    Replaces the first `old` in the file at `path` under the copy with
    `new`, failing where the file holds no `old`, so a case cannot pass on
    an edit that never landed.
    """
    file = checkout.root / path
    text = file.read_text(encoding="utf-8")

    assert old in text
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def executable(path: Path, text: str) -> Path:
    """
    Writes `text` to the file at `path`, creating its folder, and marks
    it executable.

    Returns:
        The path written.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def git(checkout: Checkout, *arguments: str):
    """
    Runs git with `arguments` at the root of `checkout` under a fixed
    author, failing the case where git fails.
    """
    checkout.run(
        "git",
        "-c",
        "user.email=thermur@example.com",
        "-c",
        "user.name=Thermur",
        *arguments
    )


def hold(checkout: Checkout, holds: dict[str, str]):
    """
    Writes `holds`, each pin's reason by the pin, as the
    `[tool.thermur.holds]` table of the copy's `pyproject.toml`, in place of
    the holds it records.
    """
    manifest = checkout.root / "pyproject.toml"
    text     = sub(
        r"(?ms)^\[tool\.thermur\.holds\]\n.*?(?=^\[)",
        "",
        manifest.read_text()
    )
    manifest.write_text(
        text
        + "\n[tool.thermur.holds]\n"
        + "".join(f"{dumps(pin)} = {dumps(reason)}\n" for pin, reason in holds.items()),
        encoding = "utf-8"
    )


def task(checkout: Checkout, name: str, text: str) -> Path:
    """
    Writes an executable task file named `name` under `.mise/tasks/` in the
    copy, which mise lists only once it can run it.

    Returns:
        The task's path relative to the root.
    """
    path = Path(".mise/tasks", name)
    executable(checkout.root / path, text)
    return path
