"""
Holds `Outputs`, the files the `docs:render` task writes into `docs/`,
each read out of a declaration in the package, which `repo:audit` holds the
committed copies to.
"""

from functools import cached_property
from pathlib   import Path
from pydantic  import AfterValidator, BaseModel, Field
from typing    import Annotated

from thermur.identity.schemas import Identity


class Outputs(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The files the render task writes under `root`, each named relative to
    it.
    """

    identity: Path = Path("docs/.vitepress/lib/rendered/identity.json")
    """
    The file the site's theme and its preview cards read the identity from.
    """

    root: Annotated[Path, AfterValidator(Path.resolve)] = Field(
        default_factory = Path.cwd
    )
    """
    The repository's root, the working directory by default.
    """

    @property
    def drifted(self) -> list[Path]:
        """
        Lists each file whose committed text differs from what the render
        would write, a missing file among them.
        """
        return [
            path
            for path, text in self.written.items()
            if not (file := self.root / path).is_file()
            or file.read_text(encoding="utf-8") != text
        ]

    @cached_property
    def written(self) -> dict[Path, str]:
        """
        Renders the text of each file the render task writes, by its path,
        the identity as JSON with its fields in the order the records
        declare them.
        """
        return {self.identity: Identity.shipped().model_dump_json(indent=2) + "\n"}

    def write(self) -> list[Path]:
        """
        Writes each drifted file, leaving every current one untouched.

        Returns:
            The paths of the files it wrote, in the order `written` lists
            them.
        """
        for path in (drifted := self.drifted):
            (file := self.root / path).parent.mkdir(exist_ok=True, parents=True)
            file.write_text(self.written[path], encoding="utf-8")

        return drifted
