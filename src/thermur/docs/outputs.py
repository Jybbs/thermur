"""
Holds `Outputs`, the files the `docs:render` task writes, meaning the
identity the site reads, written into `docs/` out of a declaration in
the package, and every Markdown page git tracks, carrying the footnote
definition of each study it cites out of `docs/references.bib`, which
`repo:audit` holds the committed copies to.
"""

from functools import cached_property
from pathlib   import Path
from pydantic  import BaseModel, Field

from thermur.identity.schemas import Identity
from thermur.repo.checkout    import Checkout
from thermur.repo.schemas     import Finding


class Outputs(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The files the render task writes under the root of `checkout`, each
    named relative to it.
    """

    checkout: Checkout = Field(default_factory=Checkout)
    """
    The clone the task writes into, the working directory by default.
    """

    identity: Path = Path("docs/.vitepress/lib/rendered/identity.json")
    """
    The file the site's theme and its preview cards read the identity from.
    """

    @property
    def drifted(self) -> list[Path]:
        """
        Lists each file whose text on disk differs from what the render
        would write, a missing file among them.
        """
        return [
            path
            for path, text in self.written.items()
            if not (file := self.checkout.root / path).is_file()
            or file.read_text(encoding="utf-8") != text
        ]

    @property
    def unknown(self) -> list[Finding]:
        """
        Names each marker a page carries under a study's key that the
        bibliography holds no entry for, page by page in path order, at the
        line of the block it sits in.
        """
        bibliography = self.checkout.bibliography
        return [
            Finding(
                file    = path,
                message = f"Line {line} of `{path}` cites `{key}`, which names no "
                f"entry in `{bibliography.file}`"
            )
            for path, page in self.checkout.pages.items()
            for line, key in page.unknown(bibliography)
        ]

    @cached_property
    def written(self) -> dict[Path, str]:
        """
        Renders the text of each file the render task writes, by its path,
        the identity as JSON with its fields in the order the records
        declare them and each page with its definitions as `Page.cited`
        writes them.
        """
        return {self.identity: Identity.shipped().model_dump_json(indent=2) + "\n"} | {
            path: page.cited(self.checkout.bibliography)
            for path, page in self.checkout.pages.items()
        }

    def write(self) -> list[Path]:
        """
        Writes each drifted file, leaving every current one untouched.

        Returns:
            The paths of the files it wrote, in the order `written` lists
            them.
        """
        for path in (drifted := self.drifted):
            (file := self.checkout.root / path).parent.mkdir(
                exist_ok = True,
                parents  = True
            )
            file.write_text(self.written[path], encoding="utf-8")

        return drifted
