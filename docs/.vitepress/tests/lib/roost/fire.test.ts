import { describe, expect, it } from 'vitest'

import { FLIGHT } from '../../../lib/front/flock.ts'
import { Fire }   from '../../../lib/roost/fire.ts'

const fire = new Fire()

describe('share', () => {
  it('holds more of the anomaly nearer the middle', () => {
    expect(fire.share(150, 50)).toBeGreaterThan(fire.share(150, 400))
  })

  it('holds none well clear of the fire', () => {
    expect(fire.share(0, 5000)).toBeLessThan(FLIGHT.faint)
  })
})

describe('reach', () => {
  it('reads back the distance a share holds at, a hotter share reaching less far', () => {
    expect(fire.share(200, fire.reach(200, 0.3))).toBeCloseTo(0.3, 6)
    expect(fire.reach(200, 0.9)).toBeLessThan(fire.reach(200, 0.1))
  })

  it('reaches nowhere before the fire has spread to the share', () => {
    expect(fire.reach(0, 0.9)).toBe(0)
  })
})

describe('felt', () => {
  it('is nothing at the least heat that pushes a drone', () => {
    expect(fire.felt(FLIGHT.faint)).toBe(0)
  })

  it('reaches one once the heat reaches 0.42 of the anomaly, and holds there', () => {
    expect(fire.felt(0.42)).toBeCloseTo(1, 6)
    expect(fire.felt(1)).toBe(1)
  })

  it('rises with the heat between', () => {
    expect(fire.felt(0.05)).toBeGreaterThan(fire.felt(0.005))
  })
})

describe('push', () => {
  const positions = Float64Array.from([300, 0, 100, 0, 0, 100, 9000, 9000, 100])

  it('pushes a drone straight away from the middle, and leaves one out of the fire alone', () => {
    const pushes = new Float64Array(9)
    fire.push(250, 0, 0, positions, pushes)
    expect(pushes[0]).toBeGreaterThan(0)
    expect(pushes[1]).toBe(0)
    expect(pushes[7]).toBe(0)
    expect(pushes[6]).toBe(0)
  })

  it('writes over what the last push left', () => {
    const pushes = new Float64Array(9).fill(7)
    fire.push(250, 0, 0, positions, pushes)
    expect(pushes[2]).toBe(0)
  })

  it('pushes a drone at the middle itself without dividing by zero', () => {
    const pushes = new Float64Array(3)
    fire.push(250, 0, 0, Float64Array.from([0, 0, 100]), pushes)
    expect(Number.isFinite(pushes[0])).toBe(true)
  })
})
