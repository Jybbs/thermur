// @vitest-environment node
import path from 'node:path'

import { parseSync }                    from 'vite'
import { expect, it, vi }               from 'vitest'
import { type Component, createSSRApp } from 'vue'
import { renderToString }               from 'vue/server-renderer'

import credit from '../../../lib/rendered/credit.json' with { type: 'json' }

vi.mock('vitepress', () => ({ withBase: (to: string) => `/base${to}` }))

/**
 * Each module of the site, by its path under the site's own folder, as text.
 */
const MODULES = import.meta.glob<string>('./{lib,theme}/**/*.ts', {
  base   : '../../../',
  eager  : true,
  import : 'default',
  query  : '?raw'
})
/**
 * Each rendered file holding a value of the sample's record a figure draws.
 */
const SAMPLED = ['./lib/rendered/crop.json']
/**
 * Each module of the theme as it loads, by the path `MODULES` files it under.
 */
const WIDGETS = import.meta.glob<{ default: Component }>('./theme/**/*.ts', { base: '../../../' })

/**
 * Lists the module at `file` beside every module it imports or exports from,
 * directly or through another.
 */
function reached(file: string, seen = new Set<string>()): Set<string> {
  if (seen.has(file)) return seen
  seen.add(file)
  const { module: { staticExports, staticImports } } = parseSync(file, MODULES[file] ?? '')
  for (const { moduleRequest } of [...staticImports, ...staticExports.flatMap(({ entries }) => entries)]) {
    if (moduleRequest?.value.startsWith('.')) {
      reached(`./${path.posix.join(path.posix.dirname(file), moduleRequest.value)}`, seen)
    }
  }
  return seen
}

/**
 * Each module of the theme laying out a figure drawn from the sample's record.
 */
const figures = Object.keys(WIDGETS).filter(file =>
  MODULES[file]!.includes('h(\'figure\'') && SAMPLED.some(json => reached(file).has(json)))

it('finds every figure drawn from the sample', () => {
  expect(figures).toContain('./theme/crop.ts')
})

it('walks through a module exporting from another', () => {
  expect(reached('./theme/identity/widgets.ts')).toContain('./lib/rendered/identity.json')
})

it.each(figures)('ends the caption of %s on the link to the credits, with no script run', async (file) => {
  const html = await renderToString(createSSRApp((await WIDGETS[file]!()).default))
  const link = `<a class="credits-link" href="/base/data/#credits" title="${credit.title}">Credits ↓</a>`
  expect(html).toContain(`${link}</figcaption></figure>`)
})
