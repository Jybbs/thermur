"""
Holds the edits a case makes to its copy of the checkout before a check
reads it, meaning text added to the end of a file, one string replaced in a
file, and a task file written fresh.
"""

from pathlib import Path

from thermur.repo.checkout import Checkout


def append(checkout: Checkout, path: str, text: str):
    """
    Adds `text` to the end of the file at `path` under the copy.
    """
    with (checkout.root / path).open("a", encoding="utf-8") as file:
        file.write(text)


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


def task(checkout: Checkout, name: str, text: str) -> Path:
    """
    Writes an executable task file named `name` under `.mise/tasks/` in the
    copy, which mise lists only once it can run it.

    Returns:
        The task's path relative to the root.
    """
    path = Path(".mise/tasks", name)
    file = checkout.root / path
    file.parent.mkdir(exist_ok=True, parents=True)
    file.write_text(text, encoding="utf-8")
    file.chmod(0o755)
    return path
