import { createMarkdownRenderer, disposeMdItInstance } from 'vitepress'
import { afterAll, expect, it }                        from 'vitest'

import { markdownRules } from '../../lib/markdown.ts'

afterAll(() => {
  disposeMdItInstance()
})

it('sets the formulas, the widgets, and the footnotes on one renderer', async () => {
  const md   = await createMarkdownRenderer(process.cwd(), { config: markdownRules })
  const html = await md.renderAsync('Cited $`x`$[^a].\n\n<!-- kept -->\n\n[^a]: Noted.\n')
  expect(html).toContain('class="katex"')
  expect(html).toContain('<CropKept />')
  expect(html).toContain('<SourcesFlock />')
})
