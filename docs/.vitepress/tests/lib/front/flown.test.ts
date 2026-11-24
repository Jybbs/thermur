import { describe, expect, it } from 'vitest'

import { Flown } from '../../../lib/front/flown.ts'

/**
 * Flies `count` drones from `seed` for `steps` steps under no push, returning
 * the flight.
 */
function flown(seed: number, steps = 20, count = 40): Flown {
  const flight = new Flown(count, seed)
  flight.launch()
  for (let step = 0; step < steps; step++) flight.advance(new Float64Array(count * 3))
  return flight
}

describe('Flown', () => {
  it('flies the same flight from one seed, and another from any other', () => {
    expect([...flown(3).composite.positions]).toEqual([...flown(3).composite.positions])
    expect([...flown(3).composite.positions]).not.toEqual([...flown(4).composite.positions])
  })

  it('counts the time from its launch by whole steps', () => {
    const flight = flown(3, 25)
    expect(flight.seconds).toBeCloseTo(25 * flight.composite.model.plan.step_s, 9)
    flight.launch()
    expect(flight.seconds).toBe(0)
  })

  it('moves the flock under the push it is given', () => {
    const still  = flown(3, 0)
    const pushed = flown(3, 0)
    const push   = new Float64Array(120).fill(50)
    for (let step = 0; step < 10; step++) {
      still.advance(new Float64Array(120))
      pushed.advance(push)
    }
    expect(pushed.composite.positions[0]!).toBeGreaterThan(still.composite.positions[0]!)
  })
})
