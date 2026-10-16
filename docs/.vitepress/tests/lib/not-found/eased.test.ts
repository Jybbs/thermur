import { describe, expect, it } from 'vitest'

import { EasedPointer } from '../../../lib/not-found/eased.ts'

describe('EasedPointer', () => {
  it('starts at the pointer as it arrives, then eases toward it as it moves', () => {
    const aim = new EasedPointer()
    aim.step(0.016, [100, 50])
    expect([aim.x, aim.y]).toEqual([100, 50])
    aim.step(0.016, [200, 50])
    expect(aim.x).toBeGreaterThan(100)
    expect(aim.x).toBeLessThan(200)
  })

  it('fades in as the pointer stays and out once it leaves', () => {
    const aim = new EasedPointer()
    for (let k = 0; k < 120; k++) aim.step(0.016, [100, 50])
    expect(aim.on).toBeGreaterThan(0.99)
    for (let k = 0; k < 120; k++) aim.step(0.016, null)
    expect(aim.on).toBeLessThan(0.01)
  })

  it('stands at the pointer itself, fully on or off, while the drawing stands still', () => {
    const aim = new EasedPointer()
    expect(aim.frame({ pointer: [30, 40], still: true })).toMatchObject({ on: 1, x: 30, y: 40 })
    expect(aim.frame({ pointer: null, still: true })).toMatchObject({ on: 0, x: 30, y: 40 })
    expect(aim.frame({ pointer: [90, 90], still: false })).toMatchObject({ on: 0, x: 30, y: 40 })
  })
})
