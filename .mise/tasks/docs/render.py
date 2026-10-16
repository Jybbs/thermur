#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "render"
# MISE description = "Write the facts the site reads out of the package's declarations"
"""
Writes each file `thermur.docs.outputs` declares into `docs/`, as the
`docs:render` task, printing the path of each file it rewrote.
"""

from thermur.docs.outputs import Outputs


def main():
    """
    Writes each file whose committed text differs from what the render would
    write, then prints each path it wrote.
    """
    for path in Outputs().write():
        print(path)


if __name__ == "__main__":
    main()
