// @vitest-environment node
import { describe, expect, it }    from 'vitest'
import { defineComponent, h, ref } from 'vue'
import { renderToString }          from 'vue/server-renderer'

import { PIECES }    from '../../../lib/kit/pieces.ts'
import { useGuide }  from '../../../lib/kit/walk.ts'
import { PieceCard } from '../../../theme/kit/card.ts'
import { SCENES }    from '../../../theme/kit/scenes.ts'

describe('the kit’s catalog on the server', () => {
  it.each(Object.keys(SCENES))('renders the %s scene before any browser stands', async (key) => {
    expect(await renderToString(h(SCENES[key]!))).toContain('kit-catalog')
  })

  it.each(PIECES.map(piece => piece.key))('renders the %s card and its examples before any browser stands, its count at zero', async (shown) => {
    expect(await renderToString(h(PieceCard, { readers: {}, shown }))).toContain('<span class="kit-card-figure">0</span>')
  })

  it('renders an element the guide watches before any browser stands', async () => {
    const guided = defineComponent({
      setup() {
        const held = ref<HTMLElement>()
        useGuide(held, () => 'one')
        return () => h('div', { ref: held })
      }
    })
    expect(await renderToString(h(guided))).toBe('<div></div>')
  })
})
