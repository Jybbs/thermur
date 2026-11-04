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

  it('folds the crop’s table shut with no widget over it', async () => {
    const html = await rendered('<!-- crop -->\n\n| a |\n|---|\n| b |\n\n<!-- /crop -->\n')
    expect(html).toMatch(/^<details class="every-value"><summary>Every value as a table<\/summary>\n/u)
    expect(html.indexOf('<table')).toBeLessThan(html.indexOf('</details>'))
  })

  it('sets the crop’s widget alone where its marker stands', async () => {
    expect(await rendered('<!-- kept -->\n')).toBe('<CropKept />\n')
  })

  it.for(['<!-- sample -->', '<div>kept</div>'])('leaves %s as it was written', async block => {
    expect(await rendered(`${block}\n`)).toBe(`${block}\n`)
  })
})
