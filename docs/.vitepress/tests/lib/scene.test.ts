import { describe, expect, it } from 'vitest'

import * as scene from '../../lib/scene.ts'
import { birds }  from '../common/scene.ts'

/**
 * A flock of forty birds along a level spine, out of every plume's heat, its
 * birds black and from four to eight pixels long, which a case lays out as
 * given or with one field replaced.
 */
const FLOCK = {
  color  : () => '#000000',
  count  : 40,
  gap    : () => 0,
  glyphs : 3,
  seed   : 7,
  size   : (_: number, draw: number) => 4 + 4 * draw,
  spine  : (along: number) => [along * 100, 50] as const,
  width  : () => 10
}

/**
 * Colors a bird red wherever it sits in the left half of the flock's spine and
 * black elsewhere.
 */
const byHalf = (_: number, __: number, x: number): string => x < 50 ? '#ff0000' : '#000000'

/**
 * Keeps every bird along the first half of the ribbon and none along the
 * second.
 */
const firstHalf = (along: number): number => along < 0.5 ? 1 : 0

/**
 * Colors a bird red wherever it sits anywhere near the parting and black
 * elsewhere.
 */
const nearness = (_: number, edge: number): string => edge > 0 ? '#ff0000' : '#000000'

describe('murmuration', () => {
  it('lays out a bird for each of its count, alike on every run from one seed', () => {
    const laid = scene.murmuration(FLOCK)
    expect(birds(laid)).toHaveLength(FLOCK.count)
    expect(scene.murmuration(FLOCK)).toBe(laid)
    expect(scene.murmuration({ ...FLOCK, seed: 8 })).not.toBe(laid)
  })

  it('leaves out every bird inside the heat', () => {
    expect(scene.murmuration({ ...FLOCK, gap: () => 2 })).toBe('')
  })

  it('turns a bird at the edge of the parting away from it and lifts it', () => {
    const open = birds(scene.murmuration(FLOCK))
    const edge = birds(scene.murmuration({ ...FLOCK, gap: () => 0.5 }))
    for (const [index, bird] of edge.entries()) {
      expect(bird.turn - open[index]!.turn).toBeCloseTo(-0.4 * 180 / Math.PI, 0)
      expect(bird.y - open[index]!.y).toBeCloseTo(-3, 0)
    }
  })

  it('colors each bird by how near the parting it sits, turning none short of the edge', () => {
    const near = birds(scene.murmuration({ ...FLOCK, color: nearness, gap: () => 0.2 }))
    expect(near.map(bird => bird.fill)).toEqual(Array.from({ length: FLOCK.count }, () => '#ff0000'))
    expect(near.map(bird => bird.turn)).toEqual(birds(scene.murmuration(FLOCK)).map(bird => bird.turn))
  })

  it('keeps a bird at each place along the ribbon at the share its density sets there', () => {
    const thinned = birds(scene.murmuration({ ...FLOCK, density: firstHalf }))
    expect(thinned).toHaveLength(FLOCK.count)
    expect(Math.max(...thinned.map(bird => bird.x))).toBeLessThan(60)
    expect(scene.murmuration({ ...FLOCK, density: () => 0 })).toBe('')
  })

  it('colors each bird by where it sits on the card', () => {
    const split = birds(scene.murmuration({ ...FLOCK, color: byHalf }))
    expect(split.map(bird => bird.fill)).toEqual(split.map(bird => byHalf(0, 0, bird.x)))
  })

  it('gathers a flock along a spine of no length at its one point', () => {
    const still = birds(scene.murmuration({ ...FLOCK, spine: () => [10, 20] as const }))
    expect(new Set(still.map(bird => `${bird.x} ${bird.y}`))).toEqual(new Set(['10 20']))
  })
})

describe('skyBands', () => {
  it('fills each band between its two edges, raised wherever the lift reports', () => {
    const tones = ['#111111', '#222222']
    const level = scene.skyBands({ edges: [0, 10, 20], lift: () => 0, tones, width: 16 })
    expect(level).toBe(
        '<path d="M0 0L8 0L16 0L16 10L8 10L0 10Z" fill="#111111"/>'
      + '<path d="M0 10L8 10L16 10L16 20L8 20L0 20Z" fill="#222222"/>'
    )
    expect(scene.skyBands({ edges: [0, 10], lift: x => x / 8, tones: ['#111111'], width: 8 })).toBe(
      '<path d="M0 0L8 -1L8 9L0 10Z" fill="#111111"/>'
    )
  })
})

describe('the smaller pieces', () => {
  it('defines each traced bird under the id of its index', () => {
    expect(scene.glyphDefs(['M0 0Z', 'M1 1Z'])).toBe('<path id="g0" d="M0 0Z"/><path id="g1" d="M1 1Z"/>')
  })

  it('traces a height every six pixels and rounds each point to tenths', () => {
    expect(scene.trace(12, x => x / 7)).toBe('M0 0L6 0.9L12 1.7')
    expect(scene.tenths(1.26)).toBe(1.3)
  })
})
