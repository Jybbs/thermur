import { describe, expect, it } from 'vitest'

import type { Box }  from '../../../lib/common.ts'
import { Flock }     from '../../../lib/front/flock.ts'
import type { Heat } from '../../../lib/front/heat.ts'
import { Ring }      from '../../../lib/front/ring.ts'

/**
 * How the case's flock flies: sixty birds drawing from twenty-eight traced
 * birds, from one seed, cruising at 30 pixels a second.
 */
const FLIGHT = { count: 60, glyphs: 28, seed: 41, sep: 9, size: [4, 6.6], speed: 30 } as const
/**
 * The region the case's flock flies over, 600 by 300 pixels.
 */
const REGION: Box = { x0: 0, x1: 600, y0: 0, y1: 300 }
/**
 * Air with no heat in it anywhere.
 */
const STILL: Heat = { heat: () => 0 }

/**
 * Flies a flock of sixty birds over a region 600 by 300 pixels through `field`
 * for `seconds`, two unless given.
 */
function flown(field: Heat = STILL, seconds = 2): Flock {
  const flock = new Flock({ ...FLIGHT, roost: middle })
  flock.region(REGION)
  flock.settle(seconds, field)
  return flock
}

/**
 * Holds the roost at the middle of its region.
 */
const middle = (_: number, box: Box, out: [number, number]): void => {
  out[0] = (box.x0 + box.x1) / 2
  out[1] = (box.y0 + box.y1) / 2
}

/**
 * Counts the birds within 40 pixels of the middle of the region.
 */
const near = (flock: Flock): number =>
  [...flock.x].filter((x, i) => Math.hypot(x - 300, flock.y[i]! - 150) < 40).length

describe('Flock', () => {
  it('flies the same flock on every run from one seed', () => {
    expect([...flown().x]).toEqual([...flown().x])
  })

  it('keeps its birds inside the region it is given', () => {
    const flock = flown(STILL, 20)
    expect(Math.min(...flock.x)).toBeGreaterThanOrEqual(REGION.x0)
    expect(Math.max(...flock.x)).toBeLessThanOrEqual(REGION.x1)
    expect(Math.min(...flock.y)).toBeGreaterThanOrEqual(REGION.y0)
    expect(Math.max(...flock.y)).toBeLessThanOrEqual(REGION.y1)
  })

  it('gives each bird one of the traced birds and a length between the two given', () => {
    const flock = flown()
    expect(Math.max(...flock.glyph)).toBeLessThan(28)
    expect(Math.min(...flock.size)).toBeGreaterThanOrEqual(4)
    expect(Math.max(...flock.size)).toBeLessThanOrEqual(6.6)
  })

  it('opens a hole where a fire burns in its middle', () => {
    const fire = new Ring({ fire: 1, radius: 120, x: 300, y: 150 })
    expect(near(flown(fire))).toBeLessThan(near(flown()))
  })

  it('moves every bird with its region when the region is resized', () => {
    const flock  = flown()
    const before = flock.x[0]!
    flock.region({ x0: 0, x1: 1200, y0: 0, y1: 300 })
    expect(flock.x[0]).toBeCloseTo(before * 2, 3)
  })
})
