import type { MarkdownRenderer } from 'vitepress'

import { footnoteRules } from './footnotes.ts'
import { mathRules }     from './math.ts'
import { widgetRules }   from './widgets.ts'

/**
 * Sets the site's own rules on the renderer VitePress builds each page with:
 * the formulas set with KaTeX, the widgets and folds a page's markers name,
 * and the footnotes drawn as the page's Sources.
 */
export function markdownRules(md: MarkdownRenderer): void {
  md.use(mathRules).use(widgetRules).use(footnoteRules)
}
