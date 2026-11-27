// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import { type Flight, PLAN, flight } from '../../../lib/roost/flight.ts'
import { Player }       from '../../../lib/roost/player.ts'
import { browser }      from '../../common/browser.ts'

/**
 * Builds a player over the page's flight, with the frames running or standing
 * still for a reader who prefers reduced motion.
 */
function played(reduced = false, flown = flight()): Player {
  browser(reduced)
  return new Player(flown)
}

/**
 * A flight that keeps no frames, for a case that reads only the loop's moment.
 */
const idle = { fly: (): void => {} } as unknown as Flight

describe('step', () => {
  it('pans in over the first seconds, the camera lifted toward the whole roost and coming down', () => {
    const player = played()
    player.step(0)
    expect(player.lift).toBe(1)
    player.step(1.5)
    expect(player.lift).toBeLessThan(1)
    expect(player.lift).toBeGreaterThan(0)
    player.step(1.5)
    expect(player.lift).toBe(0)
  })

  it('flies the flight at its pace once panned in', () => {
    const player = played()
    player.step(10)
    expect(player.t).toBeCloseTo(10 * PLAN.pace, 6)
  })

  it('holds the flight at its end, then fades the drawing out and in again at the launch', () => {
    const player = played(false, idle)
    player.step(25.3)
    expect(player.t).toBe(PLAN.seconds)
    expect(player.veil).toBe(1)
    player.step(0.7)
    expect(player.veil).toBeLessThan(1)
    player.step(0.8)
    expect(player.t).toBe(0)
    expect(player.lift).toBe(1)
    expect(player.veil).toBeCloseTo(0.5, 6)
  })

  it('comes round to the start of the loop without a cut', () => {
    const player = played(false, idle)
    player.step(27.1)
    player.step(0.2)
    expect(player.t).toBeLessThan(2)
    expect(player.veil).toBe(1)
  })
})

describe('while motion stands still', () => {
  it('shows the still moment with the drawing in full and the camera down', () => {
    const player = played(true)
    expect(player.t).toBe(PLAN.still)
    expect(player.veil).toBe(1)
    expect(player.lift).toBe(0)
  })
})

describe('place', () => {
  it('writes where the drone is between two frames', () => {
    const player = played()
    player.step(8)
    const at     = new Float32Array(3)
    player.place(3, at)
    const { count, places } = player.flight
    const frame  = player.frame
    const [a, b] = [places[(frame * count + 3) * 3 + 2]!, places[((frame + 1) * count + 3) * 3 + 2]!]
    expect(at[2]).toBeGreaterThanOrEqual(Math.min(a, b) - 1e-3)
    expect(at[2]).toBeLessThanOrEqual(Math.max(a, b) + 1e-3)
  })
})

describe('read', () => {
  it('reads a row between two frames, at an offset into each', () => {
    const player = played()
    player.step(8)
    const row    = Float32Array.from({ length: 2 * player.flight.polarization.length }, (_, k) => k)
    const frame  = player.frame
    const share  = player.t * PLAN.hz - frame
    expect(player.read(row, 2, 1)).toBeCloseTo(frame * 2 + 1 + share * 2, 3)
  })
})
