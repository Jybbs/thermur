import { describe, expect, it } from 'vitest'

import * as common  from '../../../lib/common.ts'
import { launched } from '../../../lib/front/launch.ts'
import model        from '../../../lib/rendered/composite.json' with { type: 'json' }

/**
 * Launches `count` drones from the plan with the draws of `seed` and reads
 * each drone's height.
 */
function heights(count: number, seed: number): number[] {
  const positions = new Float64Array(count * 3)
  launched(common.gaussian(common.mulberry(seed)), model.plan, positions, new Float64Array(count * 3))
  return Array.from(positions.filter((_, index) => index % 3 === 2))
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
    const plain  = heights(100, 7)
    expect(Math.min(...raised)).toBeGreaterThanOrEqual(0)
    expect(raised.reduce((sum, value) => sum + value, 0) / raised.length).toBeGreaterThan(model.plan.center_m[2]! + 40)
    expect(plain.reduce((sum, value) => sum + value, 0) / plain.length).toBeCloseTo(model.plan.center_m[2]!, -1)
  })
})
