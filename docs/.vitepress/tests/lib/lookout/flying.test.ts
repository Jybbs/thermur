import { describe, expect, it } from 'vitest'

import { FLIGHT }  from '../../../lib/front/flock.ts'
import { Flown }   from '../../../lib/front/flown.ts'
import { Flying }  from '../../../lib/lookout/flying.ts'
import model       from '../../../lib/rendered/composite.json' with { type: 'json' }

describe('Flying', () => {
  it('flies the package composite from the seed it is given under no push', () => {
    const flying = new Flying(40, 2)
    const flown  = new Flown(40, 2)
    flown.launch()
    for (let step = 0; step < 10; step++) {
      flying.play(model.plan.step_s + 1e-9)
      flown.advance(new Float64Array(120))
    }
    expect([...flying.positions]).toEqual([...flown.composite.positions])
    expect(flying.seconds).toBeCloseTo(flown.seconds, 9)
  })

  it('moves on by whole steps, carrying what is left to the next frame', () => {
    const flying = new Flying(40, 2)
    flying.play(model.plan.step_s / 2)
    expect(flying.seconds).toBe(0)
    flying.play(model.plan.step_s / 2 + 1e-9)
    expect(flying.seconds).toBeCloseTo(model.plan.step_s, 9)
  })

  it('flies no more in one frame than the site’s flocks do', () => {
    const flying = new Flying(40, 2)
    flying.play(5)
    expect(flying.seconds).toBeLessThanOrEqual(FLIGHT.stretch)
    expect(flying.count).toBe(40)
  })
})
