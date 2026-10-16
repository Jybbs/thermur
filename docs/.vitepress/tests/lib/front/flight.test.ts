import { describe, expect, it } from 'vitest'

import { plan, smoothPath } from '../../../lib/front/flight.ts'
import { FIRE, geometry }   from '../../common/front.ts'

const page = geometry()

describe('plan', () => {
  it('flies the same eight paths on every run, since the flight is drawn from one seed', () => {
    const [first, second] = [plan(page, 604, FIRE), plan(page, 604, FIRE)]
    expect([second.hs, second.xs, second.ys]).toEqual([first.hs, first.xs, first.ys])
  })

  it('lands each of the eight on the ridge at the page’s foot', () => {
    const run = plan(page, 604, FIRE)
    for (const ys of run.ys) expect(ys.at(-1)).toBeGreaterThan(page.page - 120)
    expect(Math.abs(run.ridge(600) - (page.page - 70))).toBeLessThanOrEqual(12)
  })

  it('parts the eight around the fire, flying a different flight where the fire is out', () => {
    const lit = plan(page, 604, FIRE)
    expect(Math.max(...lit.hs.flatMap(row => [...row]))).toBeGreaterThan(0)
    expect(plan(page, 604, 0).xs).not.toEqual(lit.xs)
  })

  it('starts the flight 84% of the way across a narrow page, within its 8-pixel scatter', () => {
    const narrow = plan({ ...page, w: 390 }, 604, FIRE)
    expect(Math.abs(narrow.xs[0]![0]! - 390 * 0.84)).toBeLessThanOrEqual(8)
  })
})

describe('smoothPath', () => {
  it('cuts each corner twice, keeping both ends and doubling the samples on each pass', () => {
    const smoothed = smoothPath([0, 4, 0])
    expect([smoothed[0], smoothed.at(-1)]).toEqual([0, 0])
    expect(smoothed).toHaveLength(12)
    expect(Math.max(...smoothed)).toBeLessThan(4)
  })
})
