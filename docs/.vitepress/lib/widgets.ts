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
 * What each marker writes on the site, by the marker's name.
 */
const MARKERS: Readonly<Record<string, string>> = {
  '/crop'        : SHUT,
  '/faces'       : SHUT,
  '/heat'        : SHUT,
  '/marks'       : SHUT,
  '/palette'     : SHUT,
  '/pieces'      : SHUT,
  '/threat'      : SHUT,
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
  'threat'       : `<ThreatFront />\n${FOLD}`
}

/**
 * Sets each widget and each fold a page's markers name, on the site alone, so
 * the page stays bare Markdown and its tables read on GitHub, for a reader
 * without scripts, and in the site's search.
 */
export function widgetRules(md: MarkdownRenderer): void {
  const written = md.renderer.rules.html_block!
  md.renderer.rules.html_block = (tokens, index, options, env, self) => {
    const [, name] = /^<!-- (\/?\w+) -->$/u.exec(tokens[index]!.content.trim()) ?? []
    return MARKERS[name ?? ''] ?? written(tokens, index, options, env, self)
  }
}
