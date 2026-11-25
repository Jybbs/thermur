// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { h }                               from 'vue'

import { SEAL, rimOf } from '../../../lib/credits/seal.ts'
import CreditsSeal     from '../../../theme/credits/seal.ts'

enableAutoUnmount(afterEach)

describe('CreditsSeal', () => {
  it('reads the record’s credit around its rim, each arc set along its own path', () => {
    const seal  = mount(CreditsSeal)
    const words = seal.findAll('textPath')
    const { foot, top } = rimOf()
    expect(words.map(arc => arc.text())).toEqual([top, foot])
    for (const arc of words) {
      expect(arc.attributes('startOffset')).toBe('50%')
      expect(seal.find(`path[id="${arc.attributes('href')!.slice(1)}"]`).exists()).toBe(true)
    }
  })

  it('draws the flame on the ground inside its three rings, under a glow of its own', () => {
    const seal = mount(CreditsSeal)
    expect(['edge', 'hair', 'core'].map(ring => seal.get(`.credits-ring-${ring}`).attributes('r')))
      .toEqual([SEAL.edge, SEAL.hair, SEAL.core].map(String))
    expect(seal.findAll('.credits-ground, .credits-flame, .credits-flame-ember')).toHaveLength(3)
    const glow = seal.get('radialGradient')
    expect(glow.attributes('cy')).toBe(String(SEAL.flame.y))
    expect(seal.find(`path[fill="url(#${glow.attributes('id')!})"]`).exists()).toBe(true)
  })

  it('hides from a screen reader, which reads the same credit in the lines beside it', () => {
    expect(mount(CreditsSeal).attributes('aria-hidden')).toBe('true')
  })

  it('names its gradient and arcs apart from another seal’s on the page', () => {
    const page = mount({ render: () => [h(CreditsSeal), h(CreditsSeal)] })
    const ids  = page.findAll('[id]').map(node => node.attributes('id'))
    expect(new Set(ids).size).toBe(ids.length)
  })
})
