import { describe, expect, it } from 'vitest'

import { Kept, MOST_M } from '../../../lib/crop/kept.ts'
import source           from '../../../lib/rendered/crop.json' with { type: 'json' }

describe('Kept', () => {
  it('opens on the height the record reaches, keeping half of the most the menus can keep', () => {
    const kept = new Kept()
    expect(kept.reach).toBe(source.floor_m + source.spacing_m * (source.levels - 1))
    expect(kept.top.value).toBe(kept.reach)
    expect(kept.share).toBeCloseTo(0.5)
  })

  it.each([
    { east: 1, first: 0, share: 1, top: MOST_M, west: 0 },
    { east: 0.5, first: 0, share: 0.25, top: 500, west: 0 },
    { east: 1, first: 2, share: 0.25, top: 500, west: 0 },
    { east: 0.5, first: 2, share: 0.05, top: 200, west: 0 },
    { east: 1, first: 2, share: 0.05, top: 200, west: 0.5 }
  ])('keeps $share from $west to $east up to $top m from snapshot $first', ({ share, ...parts }) => {
    const kept        = new Kept()
    kept.east.target  = parts.east
    kept.first.target = parts.first
    kept.top.target   = parts.top
    kept.west.target  = parts.west
    kept.settle()
    expect(kept.share).toBeCloseTo(share)
  })

  it('moves every part on with each step', () => {
    const kept        = new Kept()
    kept.east.target  = 0.5
    kept.first.target = 2
    kept.top.target   = 200
    kept.west.target  = 0.25
    kept.step(1 / 60)
    expect(kept.east.value).toBeLessThan(1)
    expect(kept.first.value).toBeGreaterThan(0)
    expect(kept.top.value).toBeLessThan(kept.reach)
    expect(kept.west.value).toBeGreaterThan(0)
  })
})
