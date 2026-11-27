import { createMarkdownRenderer, disposeMdItInstance } from 'vitepress'
import { afterAll, describe, expect, it }              from 'vitest'

import { widgetRules } from '../../lib/widgets.ts'

/**
 * Renders `markdown` through a renderer carrying the site's widget rules, the
 * way VitePress renders a page.
 */
const rendered = async (markdown: string): Promise<string> =>
  (await createMarkdownRenderer(process.cwd(), { config: widgetRules })).renderAsync(markdown)

afterAll(() => {
  disposeMdItInstance()
})

describe('widgetRules', () => {
  it('sets the widget a region names over its table, folded shut', async () => {
    const html = await rendered('<!-- threat -->\n\n| a |\n|---|\n| b |\n\n<!-- /threat -->\n')
    expect(html).toMatch(/^<ThreatFront \/>\n<details class="every-value"><summary>Every value as a table<\/summary>\n/u)
    expect(html).toContain('<table')
    expect(html.indexOf('<table')).toBeLessThan(html.indexOf('</details>'))
  })

  it.for([
    { marker: 'faces', widget: 'IdentityFaces' },
    { marker: 'heat', widget: 'IdentityHeat' },
    { marker: 'marks', widget: 'IdentityMarks' },
    { marker: 'palette', widget: 'IdentityPalette' }
  ])('sets $widget over the $marker table, folded shut', async ({ marker, widget }) => {
    const html = await rendered(`<!-- ${marker} -->\n\n| a |\n|---|\n| b |\n\n<!-- /${marker} -->\n`)
    expect(html).toMatch(new RegExp(`^<${widget} />\n<details class="every-value">`, 'u'))
    expect(html.indexOf('<table')).toBeLessThan(html.indexOf('</details>'))
  })

  it('folds the crop’s table shut with no widget over it', async () => {
    const html = await rendered('<!-- crop -->\n\n| a |\n|---|\n| b |\n\n<!-- /crop -->\n')
    expect(html).toMatch(/^<details class="every-value"><summary>Every value as a table<\/summary>\n/u)
    expect(html.indexOf('<table')).toBeLessThan(html.indexOf('</details>'))
  })

  it.for([
    { marker: 'kept', widget: 'CropKept' },
    { marker: 'neighbors', widget: 'RulesNeighbors' },
    { marker: 'polarization', widget: 'RulesSpacing' },
    { marker: 'reach', widget: 'RulesReach' }
  ])('sets $widget alone where the $marker marker stands', async ({ marker, widget }) => {
    expect(await rendered(`<!-- ${marker} -->\n`)).toBe(`<${widget} />\n`)
  })

  it('sets the credit’s lines inside their section, the snapshots drawn in the line their marker stands in', async () => {
    const html = await rendered('<!-- credits -->\n\n- from 0 s <!-- snapshots --> to 60 s\n\n<!-- /credits -->\n')
    expect(html).toMatch(/^<CreditsSection>\n<ul>/u)
    expect(html).toContain('<li>from 0 s <CreditsSnapshots /> to 60 s</li>')
    expect(html).toMatch(/<\/CreditsSection>\n$/u)
  })

  it('leaves another comment inside a line as it was written', async () => {
    expect(await rendered('a <!-- kept --> b\n')).toBe('<p>a <!-- kept --> b</p>\n')
  })

  it.for(['<!-- sample -->', '<!-- tagline -->', '<div>kept</div>'])('leaves %s as it was written', async block => {
    expect(await rendered(`${block}\n`)).toBe(`${block}\n`)
  })
})
