#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "render"
# MISE description = "Write the identity and each page's regions and footnotes"
"""
Writes each file `thermur.docs.outputs` declares, as the `docs:render`
task, meaning the identity into `docs/`, and each region a Markdown page
carries and the footnote definition of each study it cites into that page.
It prints the path of each file it rewrote, then each region whose page does
not carry its markers once each and each marker under a study's key that
`docs/references.bib` holds no entry for.
"""

from sys import stderr

from thermur.docs.outputs import Outputs


def main():
    """
    Writes each file whose text on disk differs from what the render would
    write and prints each path it wrote, then prints each region it left as
    it stands and each marker naming no entry, exiting with status `1` where
    there is one and `0` otherwise.
    """
    outputs = Outputs()
    for path in outputs.write():
        print(path)

    for finding in (findings := outputs.refused + outputs.unknown):
        print(finding.message, file=stderr)

    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
