import { describe, expect, it } from 'vitest'

import { Ring } from '../../../lib/front/ring.ts'

describe('Ring', () => {
  it('gives off heat falling to nothing at its radius, and none while its fire is out', () => {
    const ring = new Ring({ fire: 1, radius: 50, x: 0, y: 0 })
    expect(ring.heat(0, 0)).toBeCloseTo(1.8)
    expect(ring.heat(25, 0)).toBeCloseTo(0.45)
    expect(ring.heat(60, 0)).toBe(0)
    expect(new Ring({ fire: 0, radius: 50, x: 0, y: 0 }).heat(0, 0)).toBe(0)
  })
})
