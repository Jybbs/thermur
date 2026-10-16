import { createMarkdownRenderer, disposeMdItInstance } from 'vitepress'
import { afterAll, describe, expect, it }              from 'vitest'

import { mathRules } from '../../lib/math.ts'

/**
 * Renders `markdown` through a renderer carrying the site's math rules, the way
 * VitePress renders a page.
 */
const rendered = async (markdown: string): Promise<string> =>
  (await createMarkdownRenderer(process.cwd(), { config: mathRules })).renderAsync(markdown)

afterAll(() => {
  disposeMdItInstance()
})

describe('mathRules', () => {
  it('sets an inline formula within its line and drops both dollar signs', async () => {
    const html = await rendered('where $`x^2`$ grows')
    expect(html).toContain('<span v-pre><span class="katex">')
    expect(html).not.toContain('$')
  })

  it('sets a math block on lines of its own', async () => {
    const html = await rendered('```math\n\\xi / L\n```')
    expect(html).toContain('<div v-pre><span class="katex-display">')
    expect(html).not.toContain('<pre')
  })

  it.for([
    ['a code span with a dollar sign on one side alone', 'costs $`5` today'],
    ['a code span with no dollar signs', 'run `mise press`']
  ] as const)('leaves %s as code', async ([, markdown]) => {
    expect(await rendered(markdown)).toContain('<code>')
  })

  it('leaves a block of code in another language as code', async () => {
    expect(await rendered('```toml\nx = 1\n```')).toContain('language-toml')
  })

  it('fails on a formula KaTeX cannot parse', async () => {
    await expect(rendered('$`\\frac{`$')).rejects.toThrow('KaTeX parse error')
  })
})
