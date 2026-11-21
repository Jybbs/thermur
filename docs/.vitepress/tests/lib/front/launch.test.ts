import { describe, expect, it } from 'vitest'

import * as common  from '../../../lib/common.ts'
import { launched } from '../../../lib/front/launch.ts'
import model        from '../../../lib/rendered/composite.json' with { type: 'json' }
import parity       from '../../../lib/rendered/parity.json' with { type: 'json' }

/**
 * Launches `count` drones from the plan with the draws of `seed` and reads
 * each drone's height.
 */
function heights(count: number, seed: number): number[] {
  const positions = new Float64Array(count * 3)
  launched(common.gaussian(common.mulberry(seed)), model.plan, positions, new Float64Array(count * 3))
  return Array.from(positions.filter((_, index) => index % 3 === 2))
}

/**
 * Reads the mean of `values` over their standard deviation.
 */
function standardized(values: number[]): number {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  return mean / Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
}

describe('launched', () => {
  it.for([200, 2000])(
    'holds the plan’s spacing cubed for each of %i drones inside its cloud’s ellipsoid at one deviation',
    count => {
      const positions = new Float64Array(count * 3)
      launched(common.gaussian(common.mulberry(7)), model.plan, positions, new Float64Array(count * 3))
      const deviations = [0, 1, 2].map(axis => {
        const values = positions.filter((_, index) => index % 3 === axis)
        const mean   = values.reduce((sum, value) => sum + value, 0) / count
        return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count)
      })
      const volume = 4 * Math.PI * deviations[0]! * deviations[1]! * deviations[2]! / 3
      expect(Math.cbrt(volume / count) / model.plan.spacing_m).toBeCloseTo(1, 1)
    }
  )

  it('raises a cloud of 1000 drones off the ground by its depth and leaves one of 100 at the plan’s height', () => {
    const raised = Array.from({ length: 64 }, (_, seed) => heights(1000, seed)).flat()
    const plain  = Array.from({ length: 64 }, (_, seed) => heights(100, seed)).flat()
    expect(Math.min(...raised)).toBe(0)
    expect(standardized(raised)).toBeCloseTo(model.plan.clearance_sd, 1)
    expect(plain.reduce((sum, value) => sum + value, 0) / plain.length).toBeCloseTo(model.plan.center_m[2]!, 0)
  })

  it('launches the package’s raised flock from its draws, each drone it sets on the ground climbing', () => {
    const { draws, plan, positions, velocities } = parity.launch
    const launch = { positions: new Float64Array(plan.drones * 3), velocities: new Float64Array(plan.drones * 3) }
    const next   = draws.values()
    launched(() => next.next().value!, { ...model.plan, ...plan }, launch.positions, launch.velocities)
    for (const [name, held] of Object.entries({ positions, velocities })) {
      held.flat().forEach((value, index) => expect(launch[name as keyof typeof launch][index]).toBeCloseTo(value, 3))
    }
    const grounded = Array.from(positions.keys()).filter(drone => positions[drone]![2] === 0)
    expect(Math.min(...grounded.map(drone => draws[3 * plan.drones + 3 * drone + 2]!))).toBeLessThan(0)
    expect(Math.min(...grounded.map(drone => launch.velocities[3 * drone + 2]!))).toBeGreaterThan(0)
  })
})
