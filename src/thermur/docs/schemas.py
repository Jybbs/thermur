"""
Holds the records `docs/references.bib` validates into, meaning the
`Bibliography` of every study the file holds, each an `Article` or a
`Preprint` built on `Study`, beside the `Priority` JabRef files a study
under. The `docs:render` task writes each page's footnote definitions from
them, and the `docs:cite` task prints the definitions it is asked for.
"""

from abc          import abstractmethod
from bibtexparser import parse_string
from bibtexparser.middlewares import (
    LatexDecodingMiddleware,
    NormalizeFieldKeys,
    SeparateCoAuthors,
    SplitNameParts
)
from bibtexparser.model import Entry
from enum               import StrEnum
from pydantic           import BaseModel, Field, model_validator
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec, get_default_latex_context_db
from re          import fullmatch, sub
from typing      import Annotated, ClassVar, Literal, Self
from unicodedata import normalize

from thermur.repo.schemas import Document


class Priority(StrEnum):
    """
    How strongly a study can shape the composite, as the values JabRef
    writes in its priority field.
    """

    HIGH   = "prio1"
    LOW    = "prio3"
    MEDIUM = "prio2"


class Study(BaseModel, extra="ignore", frozen=True, use_attribute_docstrings=True):
    """
    One study's entry in `docs/references.bib`, holding the fields its
    footnote definition reads beside the annotation and the priority the
    file records for it.
    """

    form: ClassVar[str] = r"[a-z]+[0-9]{4}[a-z]?"
    """
    The form a key takes, the first author's surname and the year in
    lowercase, with a letter after the year telling apart two studies one
    author published that year.
    """

    annote: Annotated[str, Field(min_length=1)]
    """
    What the study measured, how close it sits to Thermur, and how much it
    can shape the composite.
    """

    doi: str
    """
    The study's DOI, without the address it resolves through.
    """

    key: str
    """
    The label a footnote cites the study under.
    """

    priority: Priority
    """
    How strongly the study can shape the composite.
    """

    surnames: Annotated[list[str], Field(min_length=1)]
    """
    The surname of each author in the order the study names them, each with
    any particle it carries.
    """

    title: str
    """
    The title as the study prints it, with an italic span written as
    Markdown.
    """

    year: Annotated[str, Field(pattern=r"^[0-9]{4}$")]
    """
    The year the study appeared.
    """

    @property
    def authors(self) -> str:
        """
        Names the authors the way a footnote does, meaning the surname
        alone for one author, both surnames for two, and the first surname
        followed by `et al.` for three or more.
        """
        match self.surnames:
            case [only]          : return only
            case [first, second] : return f"{first} and {second}"
            case [first, *_]     : return f"{first} et al."

    @property
    def definition(self) -> str:
        """
        Writes the footnote definition citing the study, its title closing
        on a period inside the quotes unless the title ends on a question
        mark or an exclamation mark of its own.
        """
        stop = "" if self.title.endswith(("?", "!")) else "."
        return (
            f'[^{self.key}]: {self.authors} {self.year}. "{self.title}{stop}" '
            f"{self.venue}. <https://doi.org/{self.doi}>"
        )

    @property
    def stem(self) -> str:
        """
        Spells the first author's surname in lowercase ASCII letters alone,
        followed by the year, which the key opens on.
        """
        return sub(
            r"[^a-z]",
            "",
            normalize("NFKD", self.surnames[0]).lower()
        ) + self.year

    @property
    @abstractmethod
    def venue(self) -> str:
        """
        Names where the study appeared, as the definition gives it.
        """

    @model_validator(mode="after")
    def check_key(self) -> Self:
        """
        Holds the key to the first author's surname and the year.

        Raises:
            ValueError: Where the key opens on anything but `stem` or
                        carries more than one letter after it.
        """
        if not fullmatch(rf"{self.stem}[a-z]?", self.key):
            raise ValueError(
                f"names the key `{self.key}`, where its first author and year "
                f"give `{self.stem}`"
            )

        return self


class Article(Study):
    """
    A study a journal published, as an `@article` entry.
    """

    journal: str
    """
    The journal's full name.
    """

    kind: Literal["article"]
    """
    The entry's type.
    """

    pages: str
    """
    The page range, or the article number a journal gives in its place.
    """

    volume: str
    """
    The journal's volume.
    """

    number: str | None = None
    """
    The issue within the volume, where the journal numbers one.
    """

    @property
    def venue(self) -> str:
        """
        Names the journal in italics, then its volume, the issue in
        parentheses where there is one, and the pages.
        """
        issue = f"({self.number})" if self.number else ""
        return f"*{self.journal}* {self.volume}{issue}: {self.pages}"


class Preprint(Study):
    """
    A study an archive posted ahead of any review, as a `@misc` entry naming
    the archive and its identifier there, the shape arXiv's own export
    writes.
    """

    archiveprefix: str
    """
    The archive that posted the study, such as arXiv or bioRxiv.
    """

    eprint: str
    """
    The study's identifier in that archive.
    """

    kind: Literal["misc"]
    """
    The entry's type.
    """

    @property
    def venue(self) -> str:
        """
        Names the archive, marks the study as a preprint, and gives its
        identifier there.
        """
        return f"{self.archiveprefix} preprint {self.eprint}"


class Bibliography(Document, extra="forbid", title="docs/references.bib"):
    """
    Every study `docs/references.bib` holds, under the key a footnote cites
    it by.
    """

    studies: dict[str, Annotated[Article | Preprint, Field(discriminator="kind")]]
    """
    Each study, by its key.
    """

    def cite(self, *keys: str) -> str:
        """
        Writes the definition of each study `keys` names, in the order
        given, with a blank line between two of them.
        """
        return "\n\n".join(self.studies[key].definition for key in keys)

    @staticmethod
    def load(suffix: str, text: str) -> dict[str, dict[str, dict[str, object]]]:
        """
        Reads each entry of the BibTeX in `text`, whatever its suffix, into
        the fields `Study` validates. Each field key is lowercased, each
        value decoded out of the LaTeX it is written in, writing `\\emph`
        and `\\textit` as Markdown italics and a page range's two hyphens
        as an en dash, and each author named by the surname and particle
        `SplitNameParts` separates.

        Raises:
            ValueError: Naming the line each block bibtexparser could not
                        read opens on, such as a second entry under a key
                        the file already holds, which validation reports
                        against the whole document.
        """
        context = get_default_latex_context_db()
        context.add_context_category(
            "markdown",
            macros = [MacroTextSpec(name, simplify_repl="*%s*") for name in (
                "emph", "textit"
            )],
            prepend = True
        )
        library = parse_string(
            text,
            append_middleware = [
                NormalizeFieldKeys(),
                LatexDecodingMiddleware(
                    decoder = LatexNodes2Text(
                        latex_context = context,
                        math_mode     = "verbatim"
                    )
                ),
                SeparateCoAuthors(),
                SplitNameParts()
            ]
        )
        if library.failed_blocks:
            raise ValueError(
                ", ".join(
                    f"line {block.start_line + 1} holds a block bibtexparser cannot "
                    f"read ({block.error})"
                    for block in library.failed_blocks
                )
            )

        return {"studies": {entry.key: read_entry(entry) for entry in library.entries}}

    def unknown(self, *keys: str) -> list[str]:
        """
        Names each of `keys` that no entry in the file holds.

        Returns:
            One line per such key, in the order given, and an empty list where
            every key names an entry.
        """
        return [
            f"`{key}` names no entry in `{self.file}`"
            for key in keys
            if key not in self.studies
        ]


def read_entry(entry: Entry) -> dict[str, object]:
    """
    Reads the fields of one entry bibtexparser parsed, beside its key, its
    type, and the surname of each author with any particle it carries, which
    `SplitNameParts` separates.
    """
    fields = {field.key: field.value for field in entry.fields}
    return fields | {
        "key"      : entry.key,
        "kind"     : entry.entry_type,
        "surnames" : [" ".join([*name.von, *name.last]) for name in fields.get(
            "author",
            []
        )]
    }
