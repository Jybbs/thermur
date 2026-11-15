import { describe, expect, it } from 'vitest'

import { type Broadcast, Radio, SENT } from '../../../lib/front/radio.ts'
import model                           from '../../../lib/rendered/composite.json' with { type: 'json' }
import parity                          from '../../../lib/rendered/parity.json' with { type: 'json' }

/**
 * The run of the package's radio `parity` holds.
 */
const RUN = parity.radio

/**
 * The number of drones the run carries.
 */
const COUNT = RUN.flock.positions.length

/**
 * The draws each pair's loss reads at each step a packet falls, by the step.
 */
const DRAWS = new Map(RUN.packets.map((step, packet) => [step, RUN.draws[packet]!.flat()]))

/**
 * Carries the run's flock over a copy of the radio from its opening through
 * `last`, each drone flying its velocity and each packet lost where the
 * package's draw for it falls at or past the link's chance of keeping it.
 *
 * Returns:
 *   The link after `last`, and the number of draws the package read that the
 *   copy did not, or the copy read that the package did not.
 */
function carried(last: number): { link: Radio, unread: number } {
  const { forces, positions, velocities } = RUN.flock
  const link  = new Radio(COUNT, model)
  let step    = 0
  let unread  = 0
  const broadcast: Broadcast = (index, into, at) => {
    for (let axis = 0; axis < 3; axis++) {
      into[at + axis]     = positions[index]![axis]! + step * model.plan.step_s * velocities[index]![axis]!
      into[at + 3 + axis] = velocities[index]![axis]!
    }
    into[at + 6] = forces[index]![0]!
    into[at + 7] = forces[index]![1]!
  }
  link.open(broadcast)
  for (; step <= last; step++) {
    const draws = DRAWS.get(step) ?? []
    let drawn   = 0
    link.hear(broadcast, () => 1 - draws[drawn++]!)
    unread += Math.abs(draws.length - drawn)
  }
  return { link, unread }
}

describe('Radio.hear', () => {
  it.for(RUN.packets.map((step, packet) => [step, packet] as const))(
    'hears by the packet at step %i what the package’s radio hears from the same draws, within a thousandth',
    ([step, packet]) => {
      const { link, unread } = carried(step)
      const apart            = RUN.positions[packet]!.flat().flatMap((heard, pair) =>
        heard.map((value, axis) => Math.abs(link.inbox[pair * SENT + axis]! - value))
      )
      expect(unread).toBe(0)
      expect([...link.heard]).toEqual(RUN.heard[packet]!.flat())
      expect(Math.max(...apart)).toBeLessThan(1e-3)
    }
  )
})
