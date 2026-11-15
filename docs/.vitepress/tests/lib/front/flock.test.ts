import { describe, expect, it } from 'vitest'

import type { Box }      from '../../../lib/common.ts'
import { FLIGHT, Flock } from '../../../lib/front/flock.ts'
import type { Heat }     from '../../../lib/front/heat.ts'
import { Ring }          from '../../../lib/front/ring.ts'
import model             from '../../../lib/rendered/composite.json' with { type: 'json' }

/**
 * The region the case's flock is drawn in, 600 by 300 pixels.
 */
const REGION: Box = { x0: 0, x1: 600, y0: 0, y1: 300 }
/**
 * Air with no heat in it anywhere.
 */
const STILL: Heat = { heat: () => 0 }

/**
 * Flies a flock of `count` birds, sixty unless given, drawn over a region 600
 * by 300 pixels through `field` for `seconds`, two unless given.
 */
function flown(field: Heat = STILL, seconds = 2, count = 60): Flock {
  const flock = new Flock({ ...FLIGHT, count, glyphs: 28 })
  flock.region(REGION)
  flock.settle(seconds, field)
  return flock
}

/**
 * Reads the mean of `values`.
 */
const mean = (values: Float32Array): number => values.reduce((sum, value) => sum + value, 0) / values.length

/**
 * Counts the birds within 40 pixels of the middle of the region.
 */
const near = (flock: Flock): number =>
  [...flock.x].filter((x, i) => Math.hypot(x - 300, flock.y[i]! - 150) < 40).length

describe('Flock', () => {
  it('flies the same flock on every run from one seed', () => {
    expect([...flown().x]).toEqual([...flown().x])
  })

  it('holds every bird inside its region under the roost while the flock wanders off the middle', () => {
    const flock  = flown(STILL, 2, 30)
    let farthest = 0
    for (let lap = 0; lap < 6; lap++) {
      flock.settle(10, STILL)
      farthest = Math.max(farthest, Math.abs(mean(flock.x) - 300))
      expect(Math.min(...flock.x)).toBeGreaterThan(REGION.x0)
      expect(Math.max(...flock.x)).toBeLessThan(REGION.x1)
    }
    expect(farthest).toBeGreaterThan(60)
  })

  it('draws each bird no faster than the plant’s bounds, seen from the side', () => {
    const flock = flown(STILL, 20)
    flock.vx.forEach((vx, i) => {
      expect(Math.hypot(vx, flock.vy[i]!)).toBeLessThanOrEqual(
        Math.hypot(model.plant.level_m_s, model.plant.climb_m_s) * flock.scale + 1e-3
      )
    })
  })

  it('draws the plan’s launch spacing as many pixels as the flight sets', () => {
    expect(flown().scale * model.plan.spacing_m).toBeCloseTo(FLIGHT.spacing, 6)
  })

  it('reads the distance between two birds in three dimensions, in the drawing’s pixels', () => {
    const flock   = flown()
    const offsets = [0, 1, 2].map(axis => flock.positions[3 + axis]! - flock.positions[axis]!)
    expect(flock.apart(0, 1)).toBeCloseTo(Math.hypot(...offsets) * flock.scale, 6)
    expect(flock.apart(1, 0)).toBe(flock.apart(0, 1))
  })

  it('writes the nearest birds in three dimensions, nearest first, leaving the bird out', () => {
    const flock = flown(STILL, 0, 5)
    const into  = new Int32Array(3).fill(-1)
    flock.positions.set([0, 0, 0, 0, 0, 30, 10, 0, 0, 0, 0, 20, 0, 5, 0])
    flock.nearest(new Float64Array(3), 0, into)
    expect([...into]).toEqual([4, 2, 3])
  })
})

describe('Flock’s birds', () => {
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

  it('pushes its birds up and away from heat below them', () => {
    const below: Heat = { heat: (_, y) => Math.exp(-((y - 300) ** 2) / 20000) }
    expect(mean(flown(below, 4).y)).toBeLessThan(mean(flown(STILL, 4).y))
  })

  it('moves its middle with its region', () => {
    const flock  = flown()
    const before = flock.x[0]!
    flock.region({ x0: 100, x1: 700, y0: 0, y1: 300 })
    expect(flock.x[0]).toBeCloseTo(before + 100, 3)
  })

  it('flies no more than the flight’s stretch in one frame, however long the frame', () => {
    const long  = flown()
    const short = flown()
    long.step(1, STILL)
    short.step(FLIGHT.stretch, STILL)
    expect([...long.positions]).toEqual([...short.positions])
  })

  it('flies whole steps of the plan, carrying what is left of a frame to the next one', () => {
    const flock  = flown()
    const before = [...flock.positions]
    flock.step(model.plan.step_s / 2, STILL)
    expect([...flock.positions]).toEqual(before)
    flock.step(model.plan.step_s / 2, STILL)
    expect([...flock.positions]).not.toEqual(before)
  })
})
