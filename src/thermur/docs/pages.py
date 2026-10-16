"""
Holds `Page`, one Markdown file parsed the way the site parses a page, which
finds the footnotes the file carries and rewrites the definition of each
study it cites from `docs/references.bib`. The `docs:render` task runs it
over every Markdown file git tracks, so a page carries only the marker, such
as `[^cavagna2010]`, and the task writes its definition.
"""

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
        switched off, so each definition keeps the lines it spans.
        """
        return MarkdownIt("js-default", {"html": True}).use(
            footnote_plugin,
            always_match_refs = True
        ).disable("footnote_tail").parse(self.text)

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
