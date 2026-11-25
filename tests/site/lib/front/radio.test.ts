import { describe, expect, it } from 'vitest'

import { type Broadcast, Radio, SENT } from '../../../../docs/.vitepress/lib/front/radio.ts'
import model                           from '../../../../docs/.vitepress/lib/rendered/composite.json' with { type: 'json' }
import parity                          from '../../../../docs/.vitepress/lib/rendered/parity.json' with { type: 'json' }

/**
 * The run of the package's radio `parity` holds.
 */
const RUN = parity.radio

/**
 * The run of the package's radio over a flock crossing itself, losing no packet.
 */
const CROSSING = parity.crossing

/**
 * The draws each pair's loss reads at each step a packet falls, by the step.
 */
const DRAWS = new Map(RUN.packets.map((step, packet) => [step, RUN.draws[packet]!.flat()]))

/**
 * Carries `flock` over a copy of the radio from its opening through `last`,
 * each drone flying its velocity and each packet lost where the package's draw
 * for it in `draws` falls at or past the link's chance of keeping it, a packet
 * with no draw kept.
 *
 * Returns:
 *   The link after `last`, and the number of draws the package read that the
 *   copy did not, or the copy read that the package did not.
 */
function carried(
  flock : typeof RUN.flock,
  last  : number,
  draws : Map<number, number[]> = new Map()
): { link: Radio, unread: number } {
  const { forces, positions, velocities } = flock
  const link  = new Radio(positions.length, model)
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
    const drawn = draws.get(step) ?? []
    let read    = 0
    link.hear(broadcast, () => 1 - (drawn[read++] ?? 0))
    unread += Math.abs(drawn.length - read)
  }
  return { link, unread }
}

/**
 * Lists for each receiver of `link` the senders its table holds, in the order
 * of their indices.
 */
function held(link: Radio): number[][] {
  const count   = Math.sqrt(link.heard.length)
  const senders = Array.from({ length: count }, (_, sender) => sender)
  return senders.map(receiver => senders.filter(sender => link.heard[receiver * count + sender] === 1))
}

describe('Radio.hear', () => {
  it.for(RUN.packets.map((step, packet) => [step, packet] as const))(
    'hears by the packet at step %i what the package’s radio hears from the same draws, within a thousandth',
    ([step, packet]) => {
      const { link, unread } = carried(RUN.flock, step, DRAWS)
      const count            = RUN.flock.positions.length
      const tables           = RUN.held[packet]!
      const apart            = tables.flatMap((senders, receiver) => senders.flatMap((sender, slot) =>
        RUN.positions[packet]![receiver]![slot]!.map((value, axis) =>
          Math.abs(link.inbox[(receiver * count + sender) * SENT + axis]! - value))))
      expect(unread).toBe(0)
      expect(held(link)).toEqual(tables)
      expect(Math.max(...apart)).toBeLessThan(1e-3)
    }
  )

  it.for(CROSSING.packets.map((step, packet) => [step, packet] as const))(
    'holds by the packet at step %i the senders the package’s radio holds as a flock crosses itself',
    ([step, packet]) => {
      expect(held(carried(CROSSING.flock, step).link)).toEqual(CROSSING.held[packet])
    }
  )
})
