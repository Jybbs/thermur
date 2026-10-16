import { describe, expect, it } from 'vitest'

import { contours, fireground } from '../../lib/fireground.ts'
import source                   from '../../lib/rendered/identity.json' with { type: 'json' }

/**
 * A fire at the middle of a card, at its full scale, with one puff of smoke.
 */
const FIRE = { across: 1, along: 1, puffs: 1, scale: 1, seed: 53, x: 600, y: 300 }

describe('fireground', () => {
  const fire = fireground(FIRE, source)

  it('keeps every bird out of the fire and its smoke, and none is turned well clear of both', () => {
    expect(fire.gap(600, 300)).toBe(2)
    expect(fire.gap(600 + 120 + 10, 300 - 18)).toBe(2)
    expect(fire.gap(0, 0)).toBe(0)
  })

  it('reads how near the fire’s middle a point stands, from 1 there to 0 well clear of it', () => {
    expect(fire.near(600, 300)).toBe(1)
    expect(fire.near(0, 0)).toBe(0)
  })

  it('draws the scar, an ember for each of the 160 its scale sets, and a puff of smoke', () => {
    expect(fire.svg.match(/<circle /gu)).toHaveLength(160)
    expect(fire.svg.match(/<ellipse /gu)).toHaveLength(1)
    expect(fire.svg).toContain(source.palette.colors['burnt-crimson'])
  })

  it('draws the same fire on every run from its seed', () => {
    expect(fireground(FIRE, source).svg).toBe(fire.svg)
    expect(fireground({ ...FIRE, seed: 54 }, source).svg).not.toBe(fire.svg)
  })
})

describe('contours', () => {
  it('traces twelve lines across the card, every fourth brighter than the rest', () => {
    const lines = contours('#ffffff')
    expect(lines.match(/stroke-opacity="0.14"/gu)).toHaveLength(3)
    expect(lines.match(/stroke-opacity="0.08"/gu)).toHaveLength(9)
  })
})
