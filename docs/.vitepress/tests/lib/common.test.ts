import { describe, expect, it } from 'vitest'

import * as common from '../../lib/common.ts'

/**
 * Draws the first 64 numbers of the sequence `seed` starts.
 */
const draws = (seed: number): number[] => Array.from({ length: 64 }, common.mulberry(seed))

describe('apart', () => {
  it('measures the distance between two points of two arrays, each from its own place on', () => {
    expect(common.apart(1, [9, 1, 2, 3], [0, 0, 4, 6, 3], 2)).toBe(5)
  })
})

/**
 * Draws the number every one of 16 receivers draws from every one of 16 senders
 * at the step `step` under the seed `seed`.
 */
const grid = (seed: number, step: number): number[] =>
  Array.from({ length: 256 }, (_, at) => common.hashed(seed, step, at >> 4, at & 15))

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

describe('hashed', () => {
  it('draws one fixed number in [0, 1) from each seed and triple, another from any other', () => {
    const drawn = grid(41, 3)
    expect(drawn).toEqual(grid(41, 3))
    expect(drawn).not.toEqual(grid(42, 3))
    expect(drawn).not.toEqual(grid(41, 4))
    expect(new Set(drawn).size).toBe(drawn.length)
    expect(Math.min(...drawn)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...drawn)).toBeLessThan(1)
  })

  it('tells a pair from the same pair the other way around', () => {
    expect(common.hashed(0, 0, 1, 2)).not.toBe(common.hashed(0, 0, 2, 1))
  })

  it('draws near a mean of one half', () => {
    const drawn = Array.from({ length: 4000 }, (_, at) => common.hashed(7, at, 0, 1))
    expect(drawn.reduce((sum, value) => sum + value, 0) / drawn.length).toBeCloseTo(0.5, 1)
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
    for (let x = 0; x <= 1200; x += 10) expect(Math.abs(ridge(x) - 100)).toBeLessThanOrEqual(18)
  })
})

describe('smoothstep', () => {
  it.for([[-1, 0], [0.5, 0.5], [2, 1]] as const)('eases %d to %d between 0 and 1', ([x, eased]) => {
    expect(common.smoothstep(0, 1, x)).toBe(eased)
  })
})
