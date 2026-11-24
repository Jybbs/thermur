import { afterEach, describe, expect, it, vi } from 'vitest'

import { DRONES, played, spoken } from '../../../lib/lookout/flocks.ts'
import { Flying }                 from '../../../lib/lookout/flying.ts'
import { Replayed }               from '../../../lib/lookout/replayed.ts'
import model                      from '../../../lib/rendered/composite.json' with { type: 'json' }

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('DRONES', () => {
  it('starts on the plan’s own flock, between its bounds and on its step', () => {
    expect(DRONES.value).toBe(model.plan.drones)
    expect(DRONES.value).toBeGreaterThanOrEqual(DRONES.min)
    expect(DRONES.value).toBeLessThanOrEqual(DRONES.max)
    expect((DRONES.value - DRONES.min) % DRONES.step).toBe(0)
    expect(DRONES.stops).toEqual(DRONES.stops.toSorted((a, b) => a - b))
  })
})

describe('played', () => {
  it('flies the flock in the page where no flight is served', async () => {
    const flight = await played(30, null)
    expect(flight).toBeInstanceOf(Flying)
    expect(flight.count).toBe(30)
  })

  it('asks the served address for the flight of the size given and replays it', async () => {
    const fetched = vi.fn<(address: string) => Promise<Response>>(() =>
      Promise.resolve(Response.json({ drones: 1, frame_s: 0.5, positions: [1, 2, 3], velocities: [0, 0, 0] }))
    )
    vi.stubGlobal('fetch', fetched)
    const flight = await played(1, '/flight')
    expect(fetched).toHaveBeenCalledWith('/flight?drones=1')
    expect(flight).toBeInstanceOf(Replayed)
    expect(Array.from(flight.positions)).toEqual([1, 2, 3])
  })
})

describe('spoken', () => {
  it.for([
    { count: 20, said: '20 drones' },
    { count: 3000, said: '3,000 drones' },
    { count: model.plan.drones, said: `${model.plan.drones.toLocaleString('en-US')} drones, the default` }
  ])('reads $count as “$said”', ({ count, said }) => {
    expect(spoken(count)).toBe(said)
  })
})
