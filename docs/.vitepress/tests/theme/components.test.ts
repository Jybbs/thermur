// @vitest-environment node
import { describe, expect, it, vi }               from 'vitest'
import { type Component, defineComponent, h, ref } from 'vue'
import { renderToString }                         from 'vue/server-renderer'

import { PIECES }     from '../../lib/kit/pieces.ts'
import { useGuide }   from '../../lib/kit/walk.ts'
import { COMPONENTS } from '../../theme/components.ts'
import { PieceCard }  from '../../theme/kit/card.ts'
import { SCENES }     from '../../theme/kit/scenes.ts'
import MotionPause    from '../../theme/pause.ts'

vi.mock('vitepress', () => ({
  useData  : () => ({ isDark: ref(false), page: ref({ relativePath: 'index.md' }) }),
  withBase : (path: string) => path
}))

/**
 * Renders `view` the way the site's build renders a page, in an environment
 * holding no browser global at all.
 */
const rendered = (view: Component, props: Record<string, unknown> = {}): Promise<string> =>
  renderToString(h(view, props))

describe('the widgets on the server', () => {
  it.each(Object.keys(COMPONENTS))('renders %s before any browser stands', async (name) => {
    expect(await rendered(COMPONENTS[name as keyof typeof COMPONENTS])).not.toBe('')
  })

  it('opens the kit’s catalog on its first piece', async () => {
    expect(await rendered(COMPONENTS.KitCatalog)).toContain(PIECES[0]!.name)
  })

  it('renders the page’s pause before any browser stands', async () => {
    expect(await rendered(MotionPause)).toContain('motion-pause')
  })
})

describe('the kit’s catalog on the server', () => {
  it.each(Object.keys(SCENES))('renders the %s scene before any browser stands', async (key) => {
    expect(await rendered(SCENES[key]!)).toContain('kit-catalog')
  })

  it.each(PIECES.map(piece => piece.key))('renders the %s card and its examples before any browser stands, its count at zero', async (shown) => {
    expect(await rendered(PieceCard, { readers: {}, shown })).toContain('<span class="kit-card-figure">0</span>')
  })

  it('renders an element the guide watches before any browser stands', async () => {
    const guided = defineComponent({
      setup() {
        const held = ref<HTMLElement>()
        useGuide(held, () => 'one')
        return () => h('div', { ref: held })
      }
    })
    expect(await rendered(guided)).toBe('<div></div>')
  })
})
