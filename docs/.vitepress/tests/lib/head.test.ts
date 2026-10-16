import type { HeadConfig, PageData, SiteData } from 'vitepress'
import { describe, expect, it, vi }            from 'vitest'

import type { Card, Cards } from '../../lib/cards.ts'
import * as head            from '../../lib/head.ts'
import type { Section }     from '../../lib/sections.ts'
import { FACTS }            from '../common/facts.ts'
import { REGISTRY }         from '../common/sections.ts'

const SITE = { description: 'Flies a flock.', lang: 'en-US', title: 'Thermur' } as SiteData

/**
 * What the card of a page titled `A Page` inside the section `label` at `place`
 * shows.
 */
const inSection = (label: string, place: number): Card => ({ section: { label, place }, title: 'A Page' })

/**
 * Reads the content of the meta tag naming `key`, or `undefined` where the head
 * holds none.
 */
function meta(tags: HeadConfig[], key: string): string | undefined {
  const named = (attributes: Record<string, string>): boolean =>
    (attributes.property ?? attributes.name) === key
  return tags.find(([tag, attributes]) => tag === 'meta' && named(attributes))?.[1].content
}

/**
 * Builds the previews over cards whose writes a test reads back, rendering the
 * page `relativePath` names through its head, the sections looked up in
 * `sections`.
 */
function rendered(
  relativePath : string,
  title        : string,
  isNotFound   : boolean            = false,
  sections     : readonly Section[] = []
) {
  const cards    = { write: vi.fn<Cards['write']>() } as unknown as Cards
  const previews = new head.Previews(cards, FACTS, sections)
  const pageData = { isNotFound, relativePath, title } as PageData
  const tags     = previews.head({ description: `What ${title} holds.`, pageData, siteData: SITE })
  return { cards, previews, tags }
}

describe('addressOf', () => {
  it.for([
    ['index.md', ''],
    ['methods/index.md', 'methods/'],
    ['methods/alarm.md', 'methods/alarm'],
    ['reindex.md', 'reindex']
  ])('serves %s at %j', ([relativePath, address]) => {
    expect(head.addressOf(relativePath!)).toBe(address)
  })
})

describe('cardOf', () => {
  it.for([
    ['index.md', undefined, 'cards/index.png'],
    ['404.md', undefined, 'cards/index.png'],
    ['about.md', undefined, 'cards/index.png'],
    ['methods/alarm.md', { label: 'Methods', place: 2 }, 'cards/methods/alarm.png']
  ] as const)('names the card of %s', ([relativePath, section, card]) => {
    expect(head.cardOf(relativePath, section)).toBe(card)
  })
})

describe('Previews', () => {
  it('describes the front page as the site, at its own address', () => {
    const { tags } = rendered('index.md', 'Thermur')
    expect(tags).toMatchSnapshot()
  })

  it('describes every other page as an article of the site', () => {
    const { tags } = rendered('methods/alarm.md', 'The Alarm', false, REGISTRY)
    expect(meta(tags, 'og:type')).toBe('article')
    expect(tags).toContainEqual(['link', { href: 'https://example.org/methods/alarm', rel: 'canonical' }])
    expect(JSON.parse(String(tags.at(-1)?.[2]))).toMatchObject({
      '@type'     : 'Article',
      description : 'What The Alarm holds.',
      headline    : 'The Alarm',
      image       : 'https://example.org/cards/methods/alarm.png',
      isPartOf    : { '@type': 'WebSite', name: 'Thermur', url: 'https://example.org/' }
    })
  })

  it('gives the page met at a missing address no address of its own', () => {
    const { tags } = rendered('404.md', '404', true)
    expect(meta(tags, 'og:type')).toBe('website')
    expect(meta(tags, 'og:url')).toBeUndefined()
    expect(meta(tags, 'og:image')).toBe('https://example.org/cards/index.png')
    expect(new Set(tags.map(([tag]) => tag))).toEqual(new Set(['meta']))
  })

  it('writes each alt text a reader of a preview meets as published prose', () => {
    const { tags } = rendered('methods/alarm.md', 'The Alarm')
    expect(meta(tags, 'og:image:alt')).toReadAsPublished()
    expect(meta(tags, 'twitter:image:alt')).toBe(meta(tags, 'og:image:alt'))
  })

  it('writes the main card alone for pages outside every section', async () => {
    const { cards, previews } = rendered('index.md', 'Thermur')
    previews.head({
      description : 'Where the alarm fires.',
      pageData    : { relativePath: 'about.md', title: 'About' } as PageData,
      siteData    : SITE
    })
    await previews.write('/built')
    expect(cards.write).toHaveBeenCalledWith('/built', 'cards/index.png', new Map())
  })
})

describe('Previews over a registry of sections', () => {
  it('draws a section card for each page inside a section, naming the section and its place', async () => {
    const { cards, previews } = rendered('data/obtaining.md', 'A Page', false, REGISTRY)
    for (const relativePath of ['methods/alarm.md', 'methods/index.md']) {
      const pageData = { relativePath, title: 'A Page' } as PageData
      previews.head({ description: 'Its line.', pageData, siteData: SITE })
    }
    await previews.write('/built')
    expect(cards.write).toHaveBeenCalledWith('/built', 'cards/index.png', new Map([
      ['cards/data/obtaining.png', inSection('Data', 1)],
      ['cards/methods/alarm.png', inSection('Methods', 2)],
      ['cards/methods/index.png', inSection('Methods', 2)]
    ]))
  })

  it('describes a section card’s image as published prose naming the section', () => {
    const { tags } = rendered('methods/alarm.md', 'The Alarm', false, REGISTRY)
    expect(meta(tags, 'og:image:alt')).toReadAsPublished()
    expect(meta(tags, 'og:image:alt')).toMatch(/^The Alarm, in the Methods section, /u)
  })
})

describe('siteHead', () => {
  it('names each scheme’s small mark, the faces, the site, and each scheme’s ground', () => {
    expect(head.siteHead('/flock/')).toMatchSnapshot()
  })
})
