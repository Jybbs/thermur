import fs   from 'node:fs'
import path from 'node:path'

import { type MarkdownRenderer, createMarkdownRenderer } from 'vitepress'

import { docsDir, repoRoot } from './paths.ts'

type Tokens = ReturnType<MarkdownRenderer['parse']>

/**
 * Each character GitHub drops from a heading's text as it writes the heading's
 * fragment, meaning every one outside the Unicode word class of
 * https://www.unicode.org/reports/tr18/#word, a space, and a hyphen, so a
 * variation selector such as U+FE0F and a zero-width joiner stay, as the
 * README GitHub renders shows.
 */
const DROPPED = /[^\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control} -]/gu

/**
 * The name a raw `<a>` anchor gives its place in the page, which GitHub keeps
 * as a fragment beside the headings' own.
 */
const RAW_ANCHOR = /<a\s[^>]*?\b(?:id|name)="([^"]+)"/gu

/**
 * Writes a heading's text as the fragment GitHub writes for it, lowercased,
 * each space a hyphen, and every other character outside the word class
 * dropped, per https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#section-links,
 * where the anchor plugin appends `-1`, `-2`, and on to a fragment an earlier
 * heading already holds.
 */
export function githubSlug(text: string): string {
  return text.toLowerCase().replaceAll(DROPPED, '').replaceAll(' ', '-')
}

/**
 * Lists every fragment GitHub writes in the Markdown file `tokens` parses, each
 * heading's and each raw anchor's.
 */
function fragmentsOf(tokens: Tokens): Set<string> {
  return new Set(tokens.flatMap(token => [token, ...token.children ?? []]).flatMap(token => {
    if (token.type === 'heading_open') return [token.attrGet('id') ?? '']
    if (token.type.startsWith('html')) return [...token.content.matchAll(RAW_ANCHOR)].flatMap(([, name]) => name ?? [])
    return []
  }))
}

/**
 * Names each link in the Markdown files `guides`, relative to the root of the
 * repository holding the module at `metaUrl`, whose fragment names no heading
 * or raw anchor GitHub writes in the Markdown file it leads to, the file
 * holding the link where the address is a fragment alone, each beside the line
 * it sits on.
 */
export async function strayFragments(metaUrl: string, guides: string[]): Promise<string[]> {
  const root     = repoRoot(metaUrl)
  const markdown = await createMarkdownRenderer(docsDir(metaUrl), {
    anchor : { permalink: () => undefined, slugify: githubSlug },
    attrs  : false
  })
  const parse = (file: string): Tokens | undefined => fs.existsSync(path.join(root, file))
    ? markdown.parse(fs.readFileSync(path.join(root, file), 'utf8'), {})
    : undefined

  return guides.flatMap(guide => (parse(guide) ?? []).flatMap(block => (block.children ?? [])
    .filter(child => child.type === 'link_open')
    .map(child => child.attrGet('href') ?? '')
    .filter(href => href.includes('#') && !URL.canParse(href))
    .flatMap(href => {
      const [address, fragment] = href.split('#', 2) as [string, string]
      const target = address ? path.join(path.dirname(guide), decodeURIComponent(address)) : guide
      if (!target.endsWith('.md')) return []
      const tokens = parse(target)
      return tokens && fragmentsOf(tokens).has(decodeURIComponent(fragment))
        ? []
        : [`${guide}:${(block.map?.[0] ?? 0) + 1}: \`${href}\` names no heading GitHub writes in \`${target}\``]
    })))
}
