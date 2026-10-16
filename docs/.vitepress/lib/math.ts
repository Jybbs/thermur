import { renderToString }        from 'katex'
import type { MarkdownRenderer } from 'vitepress'

/**
 * Sets `tex` with KaTeX inside an element Vue leaves uncompiled, on lines of
 * its own where `display` holds and within its line otherwise, failing on a
 * formula KaTeX cannot parse.
 */
const formula = (tex: string, display: boolean): string =>
  display
    ? `<div v-pre>${renderToString(tex, { displayMode: true, throwOnError: true })}</div>\n`
    : `<span v-pre>${renderToString(tex, { throwOnError: true })}</span>`

/**
 * Sets each formula a page writes in the two forms GitHub renders as math with
 * KaTeX, as the site parses the page:
 *
 * - A ```` ```math ```` block, set on lines of its own, which the default
 *   renderer would otherwise leave as a block of code
 * - An inline `` $`...`$ ``, a code span whose neighbors close and open on a
 *   dollar sign, set within its line with both dollar signs dropped
 *
 * Each formula sits in an element Vue leaves uncompiled, since the TeX KaTeX
 * keeps in its markup for a screen reader may hold a `{{` Vue would read as an
 * interpolation. A formula KaTeX cannot parse fails the build, naming the
 * formula.
 */
export function mathRules(md: MarkdownRenderer): void {
  md.core.ruler.after('inline', 'thermur_math', state => {
    for (const token of state.tokens) {
      if (token.type === 'fence' && token.info.trim() === 'math') {
        token.type    = 'html_block'
        token.content = formula(token.content, true)
      }

      const children = token.children ?? []
      for (const [index, child] of children.entries()) {
        const before = children[index - 1]
        const after  = children[index + 1]
        if (
          child.type !== 'code_inline'
          || before?.type !== 'text' || !before.content.endsWith('$')
          || after?.type !== 'text' || !after.content.startsWith('$')
        ) continue

        before.content = before.content.slice(0, -1)
        after.content  = after.content.slice(1)
        child.type     = 'html_inline'
        child.content  = formula(child.content, false)
      }
    }
  })
}
