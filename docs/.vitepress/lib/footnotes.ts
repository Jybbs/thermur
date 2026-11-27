import type { MarkdownRenderer } from 'vitepress'

/**
 * A rule the renderer writes one token through.
 */
type Rule = NonNullable<MarkdownRenderer['renderer']['rules']['text']>

/**
 * The state a rule of the core chain rewrites the parsed page in.
 */
type State = Parameters<Parameters<MarkdownRenderer['core']['ruler']['after']>[2]>[0]

/**
 * One token of the parsed page.
 */
type Token = Parameters<Rule>[0][number]

/**
 * What the footnote plugin keeps of each footnote while a page renders, in the
 * order the page first cites each, with the times the page cites it.
 */
interface Footnotes {
  footnotes?: { list?: { count?: number }[] }
}

/**
 * The form `mise render` writes a study's definition in, as the renderer sets
 * it: the names and the year, the title in quotes, and the venue, each closing
 * on its stop, before the DOI address.
 */
const ENTRY = /^(?<names>.+? \d{4}\.) (?<title>&quot;.+&quot;) (?<venue>.+\.) $/su
/**
 * The address every DOI resolves through, which opens the link to a study.
 */
const RESOLVER = 'https://doi.org/'

/**
 * Names the return to citation `place`, counted from 0, of a source the page
 * cites `count` times: the word back for a source cited once, and otherwise
 * the citation's letter, `a` for the first and `aa` after `z`.
 */
function backOf(place: number, count: number): string {
  if (count < 2) return 'back'
  let letters = ''
  for (let left = place + 1; left > 0; left = Math.floor((left - 1) / 26)) {
    letters = String.fromCodePoint(97 + (left - 1) % 26) + letters
  }
  return letters
}

/**
 * Holds each run of markers on one line with the word before it and the stop
 * after it, inside a span that never wraps.
 */
function held(state: State, inline: Token): void {
  const children     = inline.children ?? []
  const out: Token[] = []
  for (let start = 0; start < children.length;) {
    if (children[start]!.type !== 'footnote_ref') {
      out.push(children[start++]!)
      continue
    }
    let end = start
    while (children[end]?.type === 'footnote_ref') end++
    const before = out.at(-1)
    const after  = children[end]
    const word   = before?.type === 'text' ? /\S+$/u.exec(before.content)?.[0] ?? '' : ''
    const stop   = after?.type === 'text' ? /^\S+/u.exec(after.content)?.[0] ?? '' : ''
    if (word) before!.content = before!.content.slice(0, -word.length)
    if (stop) after!.content = after!.content.slice(stop.length)
    out.push(
      raw(state, '<span class="cite-run">'),
      ...[word].filter(Boolean).map(content => text(state, content)),
      ...children.slice(start, end),
      ...[stop].filter(Boolean).map(content => text(state, content)),
      raw(state, '</span>')
    )
    start = end
  }
  inline.children = out
}

/**
 * Builds a text token of `content`, which the renderer escapes and a heading's
 * slug reads.
 */
function text(state: State, content: string): Token {
  const token   = new state.Token('text', '', 0)
  token.content = content
  return token
}

/**
 * Builds an inline token of raw markup `content`.
 */
function raw(state: State, content: string): Token {
  const token   = new state.Token('html_inline', '', 0)
  token.content = content
  return token
}

/**
 * Rewrites each page's footnotes before they render: each run of markers held
 * with its word, each source's returns moved out of its entry to a row of
 * their own, and each study's definition set as its parts.
 */
function rewritten(state: State): void {
  let source           = false
  const backs: Token[] = []
  const kept: Token[]  = []
  for (const token of state.tokens) {
    if (token.type === 'inline') {
      held(state, token)
      if (source) sectioned(state, token)
    }
    if (token.type === 'paragraph_open' && source) token.attrJoin('class', 'source-entry')
    if (token.type === 'footnote_open') source = true
    if (token.type === 'footnote_anchor') {
      backs.push(token)
      continue
    }
    if (token.type === 'footnote_close') {
      kept.push(...backs.splice(0))
      source = false
    }
    kept.push(token)
  }
  state.tokens = kept
}

/**
 * Sets a study's definition as its parts, when its tokens hold the form `mise
 * render` writes: the names and the year in bold, the title in its quotes, the
 * venue with the journal in italics, and the DOI address on its own line,
 * split into runs that break only after a slash. A definition in any other
 * form keeps its tokens whole.
 */
function sectioned(state: State, inline: Token): void {
  const children            = inline.children ?? []
  const [open, text, close] = children.slice(-3)
  if (open?.type !== 'link_open' || text?.type !== 'text' || close?.type !== 'link_close') return
  if (!open.attrGet('href')?.startsWith(RESOLVER)) return
  const prose = state.md.renderer.renderInline(children.slice(0, -3), state.md.options, state.env)
  const parts = ENTRY.exec(prose)?.groups
  if (!parts) return
  const runs = [RESOLVER, ...text.content.slice(RESOLVER.length).split(/(?<=\/)/u)]
  open.attrJoin('class', 'source-address')
  inline.children = [
    raw(state, `<span class="source-names">${parts.names}</span> `
             + `<span class="source-title">${parts.title}</span> `
             + `<span class="source-venue">${parts.venue}</span> `),
    open,
    raw(state, runs.map(run => `<span>${state.md.utils.escapeHtml(run)}</span>`).join('<wbr>')),
    close
  ]
}

/**
 * Replaces the footnote plugin's renderer rules, so each marker reads as a
 * small number in the lead hue beside a rising bird of the mark, and the
 * definitions at a page's end read as a list headed Sources, whose numbers
 * label the birds of the flock `SourcesFlock` draws along it.
 *
 * - A marker links the source by the study's key, numbered in the order the
 *   page first cites each source, and takes an id naming its citation of it
 * - An entry opens on its number, and closes on one return per citation, each
 *   leading back to its marker
 * - The page's Markdown stays as `mise render` writes it, so GitHub renders
 *   the same footnotes in its file view
 */
export function footnoteRules(md: MarkdownRenderer): void {
  const { escapeHtml } = md.utils
  const rules          = md.renderer.rules
  const keyOf          = ({ meta }: Token): string => escapeHtml(String(meta.label ?? meta.id + 1))
  md.core.ruler.after('footnoteTail', 'thermur_sources', rewritten)

  rules.footnote_ref = (tokens, index) => {
    const token    = tokens[index]!
    const [key, n] = [keyOf(token), token.meta.id + 1]
    return `<a class="cite" href="#source-${key}" id="cite-${key}-${(token.meta.subId ?? 0) + 1}" `
         + `aria-label="Source ${n}"><span class="cite-art"></span><span class="cite-number">${n}</span></a>`
  }
  rules.footnote_block_open = () => '<section class="sources" aria-labelledby="sources">\n'
                                  + '<h2 id="sources" class="sources-title">Sources</h2>\n'
                                  + '<ol class="sources-list">\n'
  rules.footnote_block_close = () => '</ol>\n<SourcesFlock />\n</section>\n'
  rules.footnote_open        = (tokens, index) => {
    const token = tokens[index]!
    return `<li id="source-${keyOf(token)}" class="source"><span class="source-number">`
         + `<span class="source-numeral">${token.meta.id + 1}</span></span><div class="source-body">\n`
  }
  rules.footnote_close  = () => '</div></li>\n'
  rules.footnote_anchor = (tokens, index, _, env: Footnotes) => {
    const token    = tokens[index]!
    const { meta } = token
    const count    = env.footnotes?.list?.[meta.id]?.count ?? 1
    const key      = keyOf(token)
    const back     = `<a class="back" href="#cite-${key}-${meta.subId + 1}" `
                   + `aria-label="Back to citation ${meta.subId + 1} of source ${meta.id + 1}">`
                   + `${backOf(meta.subId, count)}</a>`
    const opening  = meta.subId === 0 ? '<span class="source-backs">' : ''
    const closing  = meta.subId === count - 1 ? '</span>\n' : ''
    return `${opening}${back}${closing}`
  }
}
