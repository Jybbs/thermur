// @vitest-environment node
import { describe, expect, it, vi } from 'vitest'
import { type Component, h, ref }   from 'vue'
import { renderToString }           from 'vue/server-renderer'

import { PIECES }     from '../../lib/kit/pieces.ts'
import { COMPONENTS } from '../../theme/components.ts'
import FrontSky       from '../../theme/front/sky.ts'
import NotFoundPage   from '../../theme/not-found/page.ts'
import MotionPause    from '../../theme/pause.ts'

vi.mock('vitepress', () => ({
  useData  : () => ({ isDark: ref(false), page: ref({ relativePath: 'index.md' }) }),
  withBase : (path: string) => path
}))

/**
 * Renders `view` the way the site's build renders a page, in an environment
 * holding no browser global at all.
 */
const rendered = (view: Component): Promise<string> => renderToString(h(view))

describe('the widgets on the server', () => {
  it.each(Object.keys(COMPONENTS))('renders %s before any browser stands', async (name) => {
    expect(await rendered(COMPONENTS[name as keyof typeof COMPONENTS])).not.toBe('')
  })

  it('opens the kit’s catalog on its first piece', async () => {
    expect(await rendered(COMPONENTS.KitCatalog)).toContain(PIECES[0]!.name)
  })

  it.each([
    ['the page’s pause', MotionPause, 'motion-pause'],
    ['the front page’s sky', FrontSky, 'front-sky'],
    ['the missing page', NotFoundPage, 'not-found-stage']
  ] as const)('renders %s, which no page names as a widget, before any browser stands', async (_, view, held) => {
    expect(await rendered(view)).toContain(held)
  })
})
