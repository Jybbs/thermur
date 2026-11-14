import { describe, expect, it } from 'vitest'

import { Curve, pathThrough } from '../../lib/curve.ts'

/**
 * A curve along a level line 300 pixels long, through points spaced unevenly
 * along it.
 */
const level = new Curve([[0, 50], [40, 50], [300, 50]])

describe('Curve', () => {
  it('measures its length along the points it runs through', () => {
    expect(level.length).toBeCloseTo(300, 6)
  })

  it('reads a distance along it evenly however its points are spaced', () => {
    expect(level.point(0.5)[0]).toBeCloseTo(150, 6)
    expect(level.point(-1)).toEqual([0, 50])
    expect(level.point(2)).toEqual([300, 50])
  })

  it('finds a point and the unit heading there, running straight on back past its start', () => {
    const [x, y, tx, ty] = level.at(120)
    expect([x, y, tx, ty].map(value => Math.round(value * 1e6) / 1e6)).toEqual([120, 50, 1, 0])
    expect(level.at(-20).slice(0, 2).map(value => Math.round(value * 1e6) / 1e6)).toEqual([-20, 50])
  })

  it('bends smoothly through a corner rather than turning at it', () => {
    const corner       = new Curve([[0, 0], [100, 0], [100, 100]])
    const [, , tx, ty] = corner.at(corner.length / 2)
    expect(tx).toBeGreaterThan(0.3)
    expect(ty).toBeGreaterThan(0.3)
  })
})

describe('pathThrough', () => {
  it('runs one cubic through each pair of points, its handles a sixth along the neighbors', () => {
    expect(pathThrough([[0, 0], [60, 30], [120, 0]])).toBe('M0,0C10,5 40,30 60,30C80,30 110,5 120,0')
  })

  it('passes through the same points the sampled curve does', () => {
    const points = [[0, 0], [40, 80], [90, 20], [160, 60]] as const
    const ends   = [...pathThrough(points).matchAll(/ (-?[\d.]+),(-?[\d.]+)(?=C|$)/gu)]
    expect(ends.map(([, x, y]) => [Number(x), Number(y)])).toEqual(points.slice(1))
  })
})
