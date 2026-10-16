#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "cite"
# MISE description = "Print the footnote definition of each study a key names"
# [USAGE] arg "<key>..." help="The key of each study to cite, such as cavagna2010"
"""
Prints the footnote definition of each study whose key in
`docs/references.bib` it is given, as the `docs:cite` task, for pasting
beneath an issue, a pull request, or a comment, whose text no render
reaches.
"""

from sys import argv, stderr

from thermur.repo.checkout import Checkout


def main():
    """
    Prints the definition of each study the keys name, in the order given
    with a blank line between two of them, or where a key names no entry,
    prints one line for each such key and exits with status `1` without
    printing any definition.
    """
    bibliography, keys = Checkout().bibliography, argv[1:]
    if unknown := bibliography.unknown(*keys):
        print("\n".join(unknown), file=stderr)
        raise SystemExit(1)

    print(bibliography.cite(*keys))


if __name__ == "__main__":
    main()
