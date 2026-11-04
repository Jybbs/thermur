"""
Holds `Page`, one Markdown file parsed the way the site parses a page. The
`docs:render` task runs it over every Markdown file git tracks, so a page
carries only the marker of each study it cites, such as `[^cavagna2010]`,
and the pair of HTML comments bounding each region, while the task writes
each definition from `docs/references.bib` and the body of each region.
"""

from collections.abc          import Mapping
from functools                import cached_property
from markdown_it              import MarkdownIt
from markdown_it.token        import Token
from mdit_py_plugins.footnote import footnote_plugin
from pydantic                 import BaseModel
from re import fullmatch

from thermur.docs.schemas import Bibliography, Study


class Page(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One Markdown file's text, where a footnote under the key of a study
    the bibliography holds cites that study, and a footnote under any other
    label stays as the file writes it.
    """

    text: str
    """
    The file's text.
    """

    @cached_property
    def definitions(self) -> dict[int, str]:
        """
        Maps each line a footnote definition spans, counted from zero, to
        the definition's label, the blank lines the parser reads as part of
        the definition included.
        """
        return {
            line: token.meta["label"]
            for token in self.tokens
            if token.type == "footnote_reference_open"
            for line in range(*token.map)
        }

    @cached_property
    def markers(self) -> list[tuple[int, str]]:
        """
        Lists the label of every footnote marker in the order a reader meets
        them, beside the line, counted from one, of the block each sits in.
        """
        return [
            (token.map[0] + 1, child.meta["label"])
            for token in self.tokens
            if token.type == "inline"
            for child in token.children
            if child.type == "footnote_ref"
        ]

    @cached_property
    def tokens(self) -> list[Token]:
        """
        Parses the text into markdown-it's tokens through the parser's
        default rules with HTML enabled, as VitePress parses a page, and
        the port of `markdown-it-footnote`, which reads the marker and the
        definition the way the footnote plugin VitePress turns on does.

        The plugin matches every marker whether or not the text defines it,
        and the rule gathering the definitions at the end of the text is
        left out, so each definition keeps the lines it spans.
        """
        return MarkdownIt("js-default", {"html": True}).use(
            footnote_plugin,
            always_match_refs = True,
            move_to_end       = False
        ).parse(self.text)

    def bounds(self, name: str) -> tuple[int, int] | None:
        """
        Finds the body of the region named `name`, between the one opening
        marker `<!-- name -->` the text carries and the one closing marker
        `<!-- /name -->` after it.

        Returns:
            The offset of the body's first character and the offset of the
            closing marker, or `None` where the text carries either marker
            other than once, or the closing one first.
        """
        opening, closing = f"<!-- {name} -->", f"<!-- /{name} -->"
        if self.text.count(opening) != 1 or self.text.count(closing) != 1:
            return None

        start, end = self.text.index(opening) + len(opening), self.text.index(closing)
        return (start, end) if start <= end else None

    def cited(self, bibliography: Bibliography) -> str:
        """
        Rewrites the text with every definition of a study `bibliography`
        holds removed and the definition of each such study the text cites
        set at its end, in the order the text first cites each, leaving a
        text that cites and defines none of them as it stands.
        """
        keys = dict.fromkeys(
            key for _, key in self.markers if key in bibliography.studies
        )
        dropped = {
            line
            for line, key in self.definitions.items()
            if key in bibliography.studies
        }
        if not keys and not dropped:
            return self.text

        body = "\n".join(
            line
            for index, line in enumerate(self.text.split("\n"))
            if index not in dropped
        ).rstrip()
        return "\n\n".join(filter(None, [body, bibliography.cite(*keys)])) + "\n"

    def drifted(self, regions: Mapping[str, str]) -> list[str]:
        """
        Names each region of `regions` whose body in the text differs from
        the one `filled` writes, leaving out one `bounds` finds no body for.
        """
        return [
            name
            for name, body in regions.items()
            if self.bounds(name) and self.filled({name: body}) != self.text
        ]

    def filled(self, regions: Mapping[str, str]) -> str:
        """
        Rewrites the text with the body of each region `regions` names set
        to its text, a blank line on either side, leaving a region `bounds`
        finds no body for as it stands.
        """
        text = self.text
        for name, body in regions.items():
            if span := Page(text=text).bounds(name):
                text = f"{text[: span[0]]}\n\n{body}\n\n{text[span[1] :]}"

        return text

    def unknown(self, bibliography: Bibliography) -> list[tuple[int, str]]:
        """
        Lists each marker whose label takes the form of a study's key but
        names no study `bibliography` holds, as `markers` lists it.
        """
        return [
            (line, key)
            for line, key in self.markers
            if fullmatch(Study.form, key) and key not in bibliography.studies
        ]
