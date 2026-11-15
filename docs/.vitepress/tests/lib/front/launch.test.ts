import { describe, expect, it } from 'vitest'

import * as common  from '../../../lib/common.ts'
import { launched } from '../../../lib/front/launch.ts'
import model        from '../../../lib/rendered/composite.json' with { type: 'json' }

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
})
