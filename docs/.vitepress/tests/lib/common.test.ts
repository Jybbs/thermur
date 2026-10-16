import { describe, expect, it } from 'vitest'

import * as common from '../../lib/common.ts'

/**
 * Draws the first 64 numbers of the sequence `seed` starts.
 */
const draws = (seed: number): number[] => Array.from({ length: 64 }, common.mulberry(seed))

describe('mix', () => {
  it.for([[0, '#000000'], [0.5, '#808080'], [1, '#ffffff'], [-1, '#000000'], [2, '#ffffff']] as const)(
    'lays a share of %d of white over black as %s',
    ([share, hex]) => {
      expect(common.mix('#000000', '#ffffff', share)).toBe(hex)
    }
  )
})

describe('mulberry', () => {
  it('draws one fixed sequence in [0, 1) from each seed', () => {
    expect(draws(41)).toEqual(draws(41))
    expect(draws(41)).not.toEqual(draws(42))
    expect(Math.min(...draws(41))).toBeGreaterThanOrEqual(0)
    expect(Math.max(...draws(41))).toBeLessThan(1)
  })
})

describe('nearest', () => {
  it('writes the nearest neighbors from its base on, nearest first, leaving the point itself out', () => {
    const into = new Int32Array(5).fill(-1)
    common.nearest(new Float32Array(3), 0, into, [0, 30, 10, 20, 100], [0, 0, 0, 0, 0], 2)
    expect([...into]).toEqual([-1, -1, 2, 3, 1])
  })
})

describe('ridgeline', () => {
  it('stands at its base where its waves cancel and within its amplitude everywhere', () => {
    const ridge = common.ridgeline({ amplitude: 18, base: 100, periods: [140, 61, 23], phases: [0, 0, 0] })
    expect(ridge(0)).toBe(100)
    for (let x = 0; x <= 1200; x += 10) expect(Math.abs(ridge(x) - 100)).toBeLessThanOrEqual(18)
  })
})

describe('smoothstep', () => {
  it.for([[-1, 0], [0.5, 0.5], [2, 1]] as const)('eases %d to %d between 0 and 1', ([x, eased]) => {
    expect(common.smoothstep(0, 1, x)).toBe(eased)
  })
})
