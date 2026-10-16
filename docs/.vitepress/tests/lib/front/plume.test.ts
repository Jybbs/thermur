import { describe, expect, it } from 'vitest'

import { Plume }  from '../../../lib/front/plume.ts'
import { pushed } from '../../common/front.ts'

describe('Plume', () => {
  const rise  = { core: 22, gain: 1.7, lean: 0.42, reach: 300, spread: 0.24 }
  const plume = Object.assign(new Plume(rise), { fire: 0.62, x: 400, y: 600 })

  it('gives off heat above its fire, leaning downwind as it rises, and none below the ground', () => {
    expect(plume.heat(400, 590)).toBeGreaterThan(plume.heat(300, 590))
    expect(plume.heat(484, 400)).toBeGreaterThan(plume.heat(400, 400))
    expect(plume.heat(400, 700)).toBe(0)
  })

  it('gives off no heat while its fire is out', () => {
    expect(Object.assign(new Plume(plume.rise), { fire: 0, x: 400, y: 600 }).heat(400, 590)).toBe(0)
  })

  it('pushes a flock outward from its core', () => {
    expect(pushed(plume, 380, 590)[0]).toBeLessThan(0)
  })
})
