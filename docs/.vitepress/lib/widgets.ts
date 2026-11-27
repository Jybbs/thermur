import type { MarkdownRenderer } from 'vitepress'

/**
 * What opens the fold each widget keeps its table in.
 */
const FOLD = '<details class="every-value"><summary>Every value as a table</summary>\n'
/**
 * What shuts a fold.
 */
const SHUT = '</details>\n'
/**
 * What each marker inside a line writes on the site, by the marker's name.
 */
const INLINE: Readonly<Record<string, string>> = { snapshots: '<CreditsSnapshots />' }
/**
 * What each marker standing alone writes on the site, by the marker's name.
 */
const MARKERS: Readonly<Record<string, string>> = {
  '/credits'     : '</CreditsSection>\n',
  '/crop'        : SHUT,
  '/faces'       : SHUT,
  '/heat'        : SHUT,
  '/marks'       : SHUT,
  '/palette'     : SHUT,
  '/pieces'      : SHUT,
  '/threat'      : SHUT,
  'credits'      : '<CreditsSection>\n',
  'crop'         : FOLD,
  'faces'        : `<IdentityFaces />\n${FOLD}`,
  'heat'         : `<IdentityHeat />\n${FOLD}`,
  'kept'         : '<CropKept />\n',
  'marks'        : `<IdentityMarks />\n${FOLD}`,
  'neighbors'    : '<RulesNeighbors />\n',
  'palette'      : `<IdentityPalette />\n${FOLD}`,
  'pieces'       : `<KitCatalog :readers="$frontmatter.kit" />\n${FOLD}`,
  'polarization' : '<RulesSpacing />\n',
  'reach'        : '<RulesReach />\n',
  'roost'        : '<RoostView />\n',
  'threat'       : `<ThreatFront />\n${FOLD}`
}

/**
 * Sets each widget and each fold a page's markers name, on the site alone, so
 * the page stays bare Markdown and its tables read on GitHub, for a reader
 * without scripts, and in the site's search. A marker standing alone in its
 * block takes what `MARKERS` writes, and one inside a line what `INLINE` does.
 */
export function widgetRules(md: MarkdownRenderer): void {
  for (const [rule, markers] of [['html_block', MARKERS], ['html_inline', INLINE]] as const) {
    const written = md.renderer.rules[rule]!
    md.renderer.rules[rule] = (tokens, index, options, env, self) => {
      const [, name] = /^<!-- (\/?\w+) -->$/u.exec(tokens[index]!.content.trim()) ?? []
      return markers[name ?? ''] ?? written(tokens, index, options, env, self)
    }
  }
}
