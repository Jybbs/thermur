// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import { FUELS, Front, GROUND, SPAN, WINDS, reachOf } from '../../../lib/kit/front.ts'

describe('reachOf', () => {
  it('spreads at the fuel’s rate scaled by the wind’s push, stopping at the ground’s end', () => {
    expect(reachOf(1, 1, 10)).toBe(FUELS[1].rate * 10)
    expect(reachOf(2, 0, 40)).toBeCloseTo(0.3 * 0.5 * 40)
    expect(reachOf(0, WINDS.length - 1, SPAN)).toBe(FUELS[0].rate * 2 * SPAN)
    expect(reachOf(0, WINDS.length - 1, 2 * SPAN)).toBe(GROUND)
  })
})

describe('Front', () => {
  it('takes a gradient of its own, so two fronts on one page paint apart', () => {
    const ids = [new Front(), new Front()].map(front => front.node.querySelector('linearGradient')!.id)
    expect(new Set(ids).size).toBe(2)
  })

  it('rules the ground every fifty meters across its length', () => {
    const labels = [...new Front().node.querySelectorAll('text')].map(text => text.textContent)
    expect(labels).toEqual(['0 m', '50 m', '100 m', '150 m', '200 m', '250 m', '300 m', 'MADE-UP SPREAD RATES'])
  })

  it('moves the burn, the edge, and the flock to the reach it draws, the flock stopping short of the end', () => {
    const front = new Front()
    front.draw(150)
    expect(front.burn.getAttribute('width')).toBe(String(150 * front.scale))
    expect(front.edge.getAttribute('x1')).toBe(String(front.xOf(150)))
    expect(front.flock.getAttribute('transform')).toBe(`translate(${front.xOf(150)} 0)`)
    front.draw(GROUND)
    expect(front.flock.getAttribute('transform')).toBe(`translate(${front.xOf(GROUND - 58)} 0)`)
  })

  it('lays itself out at the width it stands at, writing every other ruler label where two would crowd', () => {
    const front  = new Front()
    const shown  = (): (string | null)[] => front.labels.map(label => label.getAttribute('display'))
    front.draw(150)
    expect(shown().every(display => display === 'inline')).toBe(true)
    front.fit(300)
    expect(front.node.getAttribute('viewBox')).toBe('0 0 300 112')
    expect(shown()).toEqual(['inline', 'none', 'inline', 'none', 'inline', 'none', 'inline'])
    expect(front.labels[0]!.getAttribute('font-size')).toBe('10')
    expect(front.burn.getAttribute('width')).toBe(String(150 * front.scale))
    expect(front.birds[0]!.getAttribute('cx')).toBe(String(24 * front.scale))
  })
})
