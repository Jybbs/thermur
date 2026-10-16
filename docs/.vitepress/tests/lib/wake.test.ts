import { expect, it } from 'vitest'

import { Curve }  from '../../lib/curve.ts'
import { wakeOf } from '../../lib/wake.ts'
import { birds }  from '../common/scene.ts'

/**
 * A wake along a level route 1000 pixels long, under no fire, with a box
 * standing across its route, its birds mixing white into black.
 */
const WAKE = {
  boxes   : [{ x0: 300, x1: 360, y0: 0, y1: 630 }],
  chevron : 7,
  cream   : '#000000',
  fire    : { gap: () => 0, near: () => 0, svg: '' },
  glyphs  : 3,
  lift    : '#ffffff',
  route   : new Curve([[100, 300], [600, 300], [1100, 300]]),
  rows    : 10
}

/**
 * Reports whether a bird falls inside the fixture's box or within the six
 * pixels the wake keeps clear around it.
 */
const inBox = ({ x }: { x: number }): boolean => x > 294 && x < 366

it('keeps every bird out of each box the words take', () => {
  const laid = birds(wakeOf(WAKE))
  expect(laid.length).toBeGreaterThan(100)
  expect(laid.filter(bird => inBox(bird))).toEqual([])
})

it('leaves out every scattered bird that would sit off the card', () => {
  const route = new Curve([[100, 300], [600, 300], [2400, 300]])
  expect(Math.max(...birds(wakeOf({ ...WAKE, route })).map(({ x }) => x))).toBeLessThanOrEqual(1210)
})

it('leaves the tip in the fire’s color and cools to cream as it runs on', () => {
  const laid = birds(wakeOf(WAKE))
  const warm = laid.filter(({ x }) => x < 110).map(({ fill }) => Number.parseInt(fill.slice(1, 3), 16))
  const cool = laid.filter(({ x }) => x > 700).map(({ fill }) => Number.parseInt(fill.slice(1, 3), 16))
  expect(Math.min(...warm)).toBeGreaterThan(200)
  expect(Math.max(...cool)).toBeLessThan(120)
})
