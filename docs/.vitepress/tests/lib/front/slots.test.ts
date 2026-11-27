import { describe, expect, it } from 'vitest'

import { home } from '../../../lib/front/slots.ts'
import model    from '../../../lib/rendered/composite.json' with { type: 'json' }

/**
 * The roost's pull on a level drone at the center of its disk, under a band
 * 4 m deep about 100 m and a stiffness of 0.25 per square second, as the
 * package's test of `Home` sets them.
 */
describe('home', () => {
  it.for([[100, 0], [102, 0], [98, 0], [106, -1], [94, 1]] as const)(
    'pulls a level drone at %d m back to the band by %d m/s²',
    ([height, expected]) => {
      const into = new Float64Array(3)
      home(
        0,
        new Float64Array([10, 0, 0]),
        into,
        { center_m: [1000, 1000], height_m: 100, radius_m: 1000 },
        new Float64Array([1000, 1000, height]),
        { ...model.roost, band_m: 4, stiffness_per_s2: 0.25 }
      )
      expect(into[2]).toBeCloseTo(expected, 9)
    }
  )
})
