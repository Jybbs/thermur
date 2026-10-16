import { describe, expect, it } from 'vitest'

import { PointerHeat } from '../../../lib/front/pointer-heat.ts'
import { pushed }      from '../../common/front.ts'

describe('PointerHeat', () => {
  it('gives off heat around the pointer while it is over the sky, and none otherwise', () => {
    const falcon = Object.assign(new PointerHeat(80), { x: 100, y: 100 })
    expect(falcon.heat(100, 100)).toBe(0)
    falcon.on = true
    expect(falcon.heat(100, 100)).toBeCloseTo(0.9)
    expect(falcon.heat(140, 100)).toBeCloseTo(0.9 * Math.exp(-0.5))
    expect(pushed(falcon, 90, 100)[0]).toBeLessThan(0)
  })
})
