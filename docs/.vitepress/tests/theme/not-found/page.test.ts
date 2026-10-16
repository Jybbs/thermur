// @vitest-environment happy-dom
import { mount }                    from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { h }                        from 'vue'
import { renderToString }           from 'vue/server-renderer'

import { WORDS }             from '../../../lib/not-found/words.ts'
import NotFoundPage          from '../../../theme/not-found/page.ts'
import { RELEASED, browser } from '../../common/browser.ts'

vi.mock('vitepress', () => ({ withBase: (path: string) => `/flock${path}` }))
vi.mock('../../../lib/sections.ts', async original => {
  const { REGISTRY } = await import('../../common/sections.ts')
  return { ...await original<typeof import('../../../lib/sections.ts')>(), SECTIONS: REGISTRY }
})

/**
 * Reads the value of `attribute` on each element in the document `selector`
 * finds.
 */
const attributes = (selector: string, attribute: string): (string | null)[] =>
  [...document.querySelectorAll(selector)].map(node => node.getAttribute(attribute))

/**
 * Reads the text of each element in the document `selector` finds.
 */
const read = (selector: string): string[] =>
  [...document.querySelectorAll(selector)].map(node => node.textContent ?? '')

describe('NotFoundPage', () => {
  it('sets its words and the way through the site before it mounts', async () => {
    document.body.innerHTML = await renderToString(h(NotFoundPage))
    expect(read('h1')).toEqual([WORDS.title])
    expect(read('.not-found-lead')).toEqual([WORDS.lead])
    expect(read('.not-found-track b')).toEqual(['The front page', 'Data', 'Methods'])
    expect(read('.not-found-ring')).toEqual(['', '01', '02'])
    expect(attributes('.not-found-track a', 'href')).toEqual(['/flock/', '/flock/data/', '/flock/methods/'])
    expect(attributes('.not-found-track li', 'style')).toEqual([
      null, '--hue:var(--th-series-2);', '--hue:var(--th-series-3);'
    ])
    expect(attributes('.not-found-start .not-found-ring', 'aria-hidden')).toEqual(['true'])
  })

  it('draws the murmuration once it mounts and leaves nothing behind once it unmounts', () => {
    const stand = browser()
    const page  = mount(NotFoundPage, { attachTo: document.body })
    stand.see(true)
    stand.frames.run(3)
    expect(page.find('.not-found-stage[aria-hidden="true"] canvas').exists()).toBe(true)
    page.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})
