import { describe, expect, it } from 'vitest'

import * as common                       from '../../../lib/common.ts'
import { Composite, type Model, ranked } from '../../../lib/front/composite.ts'
import model                             from '../../../lib/rendered/composite.json' with { type: 'json' }
import parity                            from '../../../lib/rendered/parity.json' with { type: 'json' }

/**
 * Builds a copy of `parity`'s flock, launched where the package launched it.
 */
function launched(): Composite {
  const copy = new Composite(parity.positions.length)
  copy.positions.set(parity.positions.flat())
  copy.velocities.set(parity.velocities.flat())
  return copy
}

/**
 * Reads the largest gap between a value `held` holds and the package's value at
 * the same place of `stepped`.
 */
const divergence = (held: Float64Array, stepped: number[][]): number =>
  Math.max(...stepped.flat().map((value, index) => Math.abs(held[index]! - value)))

/**
 * Builds a push of 50 meters per second squared on the first drone along
 * `axis`, and none on any other drone.
 */
function pushed(axis: number, length: number): Float64Array {
  const push = new Float64Array(length)
  push[axis] = 50
  return push
}

describe('Composite.advance', () => {
  it('steps the flock the package steps from one state to the package’s flock, within a thousandth', () => {
    const copy = launched()
    copy.advance(parity.increments.flat(), new Float64Array(copy.positions.length))
    expect(divergence(copy.positions, parity.stepped.positions)).toBeLessThan(1e-3)
    expect(divergence(copy.velocities, parity.stepped.velocities)).toBeLessThan(1e-3)
  })

  it('turns a drone across its flight under a push along it no further than the push across it', () => {
    const along  = launched()
    const across = launched()
    const still  = new Float64Array(along.positions.length)
    along.advance(still, pushed(0, still.length))
    across.advance(still, pushed(1, still.length))
    const quiet = launched()
    quiet.advance(still, still)
    expect(along.velocities[0]).toBeCloseTo(quiet.velocities[0]!, 1)
    expect(Math.abs(across.velocities[1]! - quiet.velocities[1]!)).toBeGreaterThan(0.5)
  })

  it('holds every drone at the cruise speed after a step', () => {
    const copy = launched()
    copy.advance(parity.increments.flat(), new Float64Array(copy.positions.length))
    for (let at = 0; at < copy.velocities.length; at += 3) {
      expect(Math.hypot(...copy.velocities.subarray(at, at + 3))).toBeCloseTo(model.plan.cruise_m_s, 6)
    }
  })

})

describe('Composite.launch', () => {
  it('launches each drone at the cruise speed within the scatter of the course', () => {
    const copy = new Composite(200)
    copy.launch(common.gaussian(common.mulberry(3)))
    const course = model.plan.course_deg * Math.PI / 180
    for (let at = 0; at < copy.velocities.length; at += 3) {
      const [east, north] = [copy.velocities[at]!, copy.velocities[at + 1]!]
      expect(Math.hypot(...copy.velocities.subarray(at, at + 3))).toBeCloseTo(model.plan.cruise_m_s, 6)
      expect(Math.cos(course) * east + Math.sin(course) * north).toBeGreaterThan(model.plan.cruise_m_s * 0.8)
    }
  })

  it('launches a cloud longer along the course than across it and thinnest in height', () => {
    const copy   = new Composite(400)
    copy.launch(common.gaussian(common.mulberry(5)))
    const spread = (axis: number): number => {
      const values = copy.positions.filter((_, index) => index % 3 === axis)
      const mean   = values.reduce((sum, value) => sum + value, 0) / values.length
      return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
    }
    expect(spread(0)).toBeGreaterThan(spread(1))
    expect(spread(1)).toBeGreaterThan(spread(2))
  })

})

describe('Composite', () => {
  it.for(['cohesion', 'contact', 'coupling', 'heading', 'noise'] as const)(
    'refuses a model whose %s slot holds a component the copy does not fly',
    slot => {
      const other = { ...model, [slot]: { ...model[slot], kind: 'Other' } } as Model
      expect(() => new Composite(4, other)).toThrow(`The ${slot} slot holds Other`)
    }
  )
})

describe('ranked', () => {
  it('writes the nearest drones in three dimensions from its base on, nearest first, leaving the drone out', () => {
    const into   = new Int32Array(4).fill(-1)
    const points = new Float64Array([0, 0, 0, 0, 0, 30, 10, 0, 0, 0, 0, 20, 0, 5, 0])
    ranked(new Float64Array(3), 0, into, points, 1)
    expect([...into]).toEqual([-1, 4, 2, 3])
  })
})
