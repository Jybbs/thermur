import { describe, expect, it } from 'vitest'

import * as common from '../../lib/common.ts'

/**
 * Draws the first 64 numbers of the sequence `seed` starts.
 */
const draws = (seed: number): number[] => Array.from({ length: 64 }, common.mulberry(seed))

describe('gaussian', () => {
  it('draws one fixed sequence from each seed, near a mean of 0 and a deviation of 1', () => {
    const drawn = Array.from({ length: 4000 }, common.gaussian(common.mulberry(41)))
    const mean  = drawn.reduce((sum, value) => sum + value, 0) / drawn.length
    const share = drawn.reduce((sum, value) => sum + (value - mean) ** 2, 0) / drawn.length
    expect(drawn.slice(0, 9)).toEqual(Array.from({ length: 9 }, common.gaussian(common.mulberry(41))))
    expect(Math.abs(mean)).toBeLessThan(0.05)
    expect(Math.sqrt(share)).toBeCloseTo(1, 1)
  })
})

describe('lerp', () => {
  it.each([[0, 2], [0.25, 3], [1, 6], [1.5, 8]])('reads the value %d of the way from 2 to 6 as %d', (share, value) => {
    expect(common.lerp(2, 6, share)).toBe(value)
  })
})

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
    const columns = Array.from({ length: 121 }, (_, step) => step * 10)
    expect(columns.filter(x => Math.abs(ridge(x) - 100) > 18)).toStrictEqual([])
  })
})

describe('smoothstep', () => {
  it.for([[-1, 0], [0.5, 0.5], [2, 1]] as const)('eases %d to %d between 0 and 1', ([x, eased]) => {
    expect(common.smoothstep(0, 1, x)).toBe(eased)
  })
})
