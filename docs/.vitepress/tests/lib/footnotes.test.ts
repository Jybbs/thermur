import { Window }                                     from 'happy-dom'
import { createMarkdownRenderer, disposeMdItInstance } from 'vitepress'
import { afterAll, describe, expect, it }              from 'vitest'

import { footnoteRules } from '../../lib/footnotes.ts'

/**
 * A study's definition as `mise render` writes it.
 */
const BALLERINI = '[^ballerini2008b]: Ballerini et al. 2008. "Empirical investigation of starling flocks: a benchmark study in '
                + 'collective animal behaviour." *Animal Behaviour* 76(1): 201–215. <https://doi.org/10.1016/j.anbehav.2008.02.004>'
/**
 * A report's definition, whose venue sets no journal in italics.
 */
const CLAUSEN = '[^clausen2003]: Clausen and Jacquet 2003. "Optimized Link State Routing Protocol (OLSR)." RFC 3626. '
              + '<https://doi.org/10.17487/RFC3626>'
/**
 * A page citing the first study twice, the report once beside it, and a note
 * under a label the bibliography lacks.
 */
const PAGE = [
  'Its thin axis is vertical[^ballerini2008b]. Packets hold for 0.6 s[^clausen2003][^note], as starlings show[^ballerini2008b].',
  '',
  '[^note]: A remark of the page\'s own, with no source.',
  '',
  BALLERINI,
  '',
  CLAUSEN,
  ''
].join('\n')

/**
 * Renders `markdown` through a renderer carrying the site's footnote rules,
 * the way VitePress renders a page, into a document a case reads.
 */
async function page(markdown: string): Promise<Document> {
  const html     = await (await createMarkdownRenderer(process.cwd(), { config: footnoteRules })).renderAsync(markdown)
  const document = new Window().document as unknown as Document
  document.body.innerHTML = html
  return document
}

/**
 * Reads the text a reader copies out of `element`, its spaces closed up.
 */
const spoken = (element: Element | null): string => element!.textContent.replaceAll(/\s+/gu, ' ').trim()

afterAll(() => {
  disposeMdItInstance()
})

describe('the markers', () => {
  it('links every marker of one source to it by its key, numbered as the page first cites each', async () => {
    const markers = [...(await page(PAGE)).querySelectorAll('a.cite')]
    expect(markers.map(each => [each.getAttribute('href'), each.id, each.getAttribute('aria-label'), spoken(each)])).toEqual([
      ['#source-ballerini2008b', 'cite-ballerini2008b-1', 'Source 1', '1'],
      ['#source-clausen2003', 'cite-clausen2003-1', 'Source 2', '2'],
      ['#source-note', 'cite-note-1', 'Source 3', '3'],
      ['#source-ballerini2008b', 'cite-ballerini2008b-2', 'Source 1', '1']
    ])
  })

  it('holds each run of markers on one line with the word before it and the stop after it', async () => {
    const runs = [...(await page(PAGE)).querySelectorAll('.cite-run')]
    expect(runs.map(run => [run.firstChild!.textContent, run.querySelectorAll('a.cite').length, run.lastChild!.textContent]))
      .toEqual([['vertical', 1, '.'], ['s', 2, ','], ['show', 1, '.']])
  })

  it('numbers an inline footnote by its place', async () => {
    const document = await page('Said^[Inline.] here.\n')
    expect(document.querySelector('a.cite')!.id).toBe('cite-1-1')
    expect(document.querySelector('a.back')!.getAttribute('href')).toBe('#cite-1-1')
  })

  it('leaves a page with no footnotes as it was', async () => {
    expect((await page('Plain text.\n')).body.innerHTML).toBe('<p>Plain text.</p>\n')
  })
})

describe('the Sources list', () => {
  it('heads the definitions Sources in place of the rule, each entry opening on its number', async () => {
    const document = await page(PAGE)
    expect(document.querySelector('hr')).toBeNull()
    expect(spoken(document.querySelector('section.sources > h2#sources'))).toBe('Sources')
    expect([...document.querySelectorAll('li.source')].map(each => [each.id, spoken(each.querySelector('.source-number'))]))
      .toEqual([['source-ballerini2008b', '1'], ['source-clausen2003', '2'], ['source-note', '3']])
    expect(document.querySelector('section.sources > sourcesflock')).not.toBeNull()
  })

  it.for([
    ['the study', 'source-ballerini2008b', BALLERINI],
    ['the report', 'source-clausen2003', CLAUSEN]
  ] as const)('writes %s word for word as `mise render` does', async ([, id, definition]) => {
    const entry = (await page(PAGE)).querySelector(`#${id} .source-entry`)
    expect(spoken(entry)).toBe(definition.replace(/^\[\^\w+\]: /u, '').replaceAll(/[*<>]/gu, ''))
  })

  it('sets the names and the year, the title in its quotes, and the venue apart', async () => {
    const entry = (await page(PAGE)).querySelector('#source-ballerini2008b .source-entry')!
    expect(spoken(entry.querySelector('.source-names'))).toBe('Ballerini et al. 2008.')
    expect(spoken(entry.querySelector('.source-title'))).toMatch(/^"Empirical .+ behaviour\."$/u)
    expect(spoken(entry.querySelector('.source-venue em'))).toBe('Animal Behaviour')
  })

  it('links the DOI address on its own, breaking only after a slash', async () => {
    const address = (await page(PAGE)).querySelector('#source-ballerini2008b a.source-address')!
    expect(address.getAttribute('href')).toBe('https://doi.org/10.1016/j.anbehav.2008.02.004')
    expect([...address.querySelectorAll('span')].map(run => run.textContent)).toEqual(['https://doi.org/', '10.1016/', 'j.anbehav.2008.02.004'])
    expect(address.querySelectorAll('wbr')).toHaveLength(2)
  })

  it('keeps the definition under a label the bibliography lacks whole', async () => {
    const entry = (await page(PAGE)).querySelector('#source-note .source-entry')!
    expect(spoken(entry)).toBe('A remark of the page\'s own, with no source.')
    expect(entry.querySelector('.source-names')).toBeNull()
  })
})

describe('the returns', () => {
  it('closes each entry on one return per citation, lettered where the page cites the source more than once', async () => {
    const document = await page(PAGE)
    const backs    = (id: string): string[][] => [...document.querySelectorAll(`#${id} .source-backs a.back`)]
      .map(each => [each.getAttribute('href')!, each.getAttribute('aria-label')!, spoken(each)])
    expect(document.querySelectorAll('.source-entry a.back')).toHaveLength(0)
    expect(backs('source-ballerini2008b')).toEqual([
      ['#cite-ballerini2008b-1', 'Back to citation 1 of source 1', 'a'],
      ['#cite-ballerini2008b-2', 'Back to citation 2 of source 1', 'b']
    ])
    expect(backs('source-clausen2003')).toEqual([['#cite-clausen2003-1', 'Back to citation 1 of source 2', 'back']])
  })

  it('letters a source cited past twenty-six times on, as a column of a sheet is lettered', async () => {
    const markdown = `${Array.from({ length: 28 }, () => 'Cited[^x].').join(' ')}\n\n[^x]: Noted.\n`
    const letters  = [...(await page(markdown)).querySelectorAll('a.back')].map(each => each.textContent)
    expect(letters.slice(24)).toEqual(['y', 'z', 'aa', 'ab'])
  })
})
