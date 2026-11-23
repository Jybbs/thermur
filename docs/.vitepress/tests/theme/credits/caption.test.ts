// @vitest-environment node
import { createMarkdownRenderer, disposeMdItInstance } from 'vitepress'
import { afterAll, describe, expect, it, vi }          from 'vitest'

import credit            from '../../../lib/rendered/credit.json' with { type: 'json' }
import { creditCaption } from '../../../theme/credits/caption.ts'
import caption           from '../../../theme/styles/credits/caption.css?raw'
import { declared }      from '../../common/sheets.ts'

vi.mock(import('vitepress'), async original => ({ ...await original(), withBase: (to: string) => `/base${to}` }))

/**
 * The selector of the link closing a caption.
 */
const LINK = '.vp-doc .credits-link'
/**
 * The Data page as Markdown.
 */
const [PAGE] = Object.values(import.meta.glob<string>('../../../../pages/data/index.md', {
  eager  : true,
  import : 'default',
  query  : '?raw'
}))

/**
 * Renders the section closing the Data page the way VitePress renders it.
 */
async function closing(): Promise<string> {
  const html = await (await createMarkdownRenderer(process.cwd())).renderAsync(PAGE!)
  return html.slice(html.lastIndexOf('<h2 '))
}

afterAll(() => {
  disposeMdItInstance()
})

it('closes the caption’s own words on a link to the Data page’s credits, named for the record’s dataset', () => {
  const [words, link] = creditCaption('What the figure shows.').children as [string, { props: Record<string, string> }]
  expect(words).toBe('What the figure shows. ')
  expect(link.props).toMatchObject({ href: '/base/data/#credits', title: credit.title })
})

it('reaches the page’s last heading, over the lines naming the record’s dataset and its license', async () => {
  const section = await closing()
  expect(section).toMatch(/^<h2 id="credits"/u)
  expect(section).toContain(`>${credit.title}</a>`)
  expect(section).toContain(`>${credit.license}</a>`)
})

describe('the link’s sheet', () => {
  it('holds the 24 px WCAG 2.2 sets at least for a press', () => {
    expect(declared('display', LINK, caption)).toBe('inline-block')
    expect(declared('min-height', LINK, caption)).toBe('1.5rem')
  })

  it('sets it in the mono and the faintest text until it is reached', () => {
    expect(caption).toMatch(/\.vp-doc \.credits-link \{[^}]*var\(--vp-font-family-mono\)/u)
    expect(declared('color', LINK, caption)).toBe('var(--vp-c-text-3)')
    expect(declared('color', `${LINK}:is(:hover, :focus-visible, :active)`, caption)).toBe('var(--vp-c-text-1)')
  })
})
