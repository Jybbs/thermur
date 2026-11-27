import { describe, expect, it } from 'vitest'

import { Flight, PLAN, flight } from '../../../lib/roost/flight.ts'

describe('Flight', () => {
  const flown = new Flight()
  flown.fly(10)

  it('keeps each frame it has flown, the launch at frame 0', () => {
    expect(flown.polarization[10]).toBeGreaterThan(0)
    expect(flown.polarization[11]).toBe(0)
    expect(flown.places[flown.count * 3 * 10 + 2]).toBeGreaterThan(0)
  })

  it('keeps the lowest and highest height around the mean', () => {
    expect(flown.low[10]!).toBeLessThanOrEqual(flown.height[10]!)
    expect(flown.high[10]!).toBeGreaterThanOrEqual(flown.height[10]!)
  })

  it('settles the followed place onto the ringed drone', () => {
    const at = flown.led * 3
    expect(Math.abs(flown.followed[0]! - flown.places[at]!)).toBeLessThan(1e-3)
  })

  it('flies no frame twice', () => {
    const before = flown.polarization[10]
    flown.fly(5)
    flown.fly(10)
    expect(flown.polarization[10]).toBe(before)
  })

  it('flies up to the still moment, handing the thread back as it goes', async () => {
    const ready = new Flight()
    await ready.ready()
    expect(ready.polarization[PLAN.still * PLAN.hz]).toBeGreaterThan(0)
    expect(ready.polarization[PLAN.still * PLAN.hz + 2]).toBe(0)
  })
})

describe('flight', () => {
  it('is the one flight, however often it is read', () => {
    expect(flight()).toBe(flight())
  })
})
