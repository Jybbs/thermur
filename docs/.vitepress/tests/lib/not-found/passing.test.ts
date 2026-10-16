import { describe, expect, it } from 'vitest'

import { passing, rimOf } from '../../../lib/not-found/passing.ts'

/**
 * An oval 100 pixels across and 50 down from its middle at 600 and 400.
 */
const OVAL = { a: 100, b: 50, cx: 600, cy: 400 }

describe('passing', () => {
  it.for([340, 380, 420, 460])('keeps a bird far upstream on its own line, %d pixels down', far => {
    expect(passing(-100_000, far, OVAL)).toBeCloseTo(far, 1)
  })

  it.for([395, 380, 420, 330])('parts the line %d pixels down around the oval at every column', far => {
    for (let x = 480; x <= 720; x += 10) {
      expect(rimOf(OVAL, x, passing(x, far, OVAL))).toBeGreaterThanOrEqual(0.999)
    }
  })

  it('keeps a line on the side of the middle it started on', () => {
    expect(passing(600, 399, OVAL)).toBeLessThan(400)
    expect(passing(600, 401, OVAL)).toBeGreaterThan(400)
  })

  it('carries the line through the middle along the oval’s rim', () => {
    expect(rimOf(OVAL, 560, passing(560, 400, OVAL))).toBeCloseTo(1, 6)
  })
})
