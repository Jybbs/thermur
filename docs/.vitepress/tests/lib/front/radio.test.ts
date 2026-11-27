import { describe, expect, it } from 'vitest'

import { type Broadcast, type Draw, Radio, SENT } from '../../../lib/front/radio.ts'
import model                                      from '../../../lib/rendered/composite.json' with { type: 'json' }
import parity                                     from '../../../lib/rendered/parity.json' with { type: 'json' }

/**
 * The run of the package's radio `parity` holds.
 */
const RUN = parity.radio

/**
 * The run of the package's radio over a flock crossing itself, losing no packet.
 */
const CROSSING = parity.crossing

/**
 * The draws each pair's loss reads at each step a packet falls, by the step,
 * each receiver's row listing every other drone in order.
 */
const DRAWS = new Map(RUN.packets.map((step, packet) => [step, RUN.draws[packet]!]))

/**
 * Carries `flock` over a copy of the radio from its opening through `last`,
 * each drone flying its velocity and each packet lost where the package's draw
 * for its pair in `draws` falls at or past the link's chance of keeping it, a
 * packet with no draw kept.
 */
function carried(flock: typeof RUN.flock, last: number, draws: Map<number, number[][]> = new Map()): Radio {
  const { forces, positions, velocities } = flock
  const link = new Radio(positions.length, model)
  let step   = 0
  const broadcast: Broadcast = (index, into, at) => {
    for (let axis = 0; axis < 3; axis++) {
      into[at + axis]     = positions[index]![axis]! + step * model.plan.step_s * velocities[index]![axis]!
      into[at + 3 + axis] = velocities[index]![axis]!
    }
    into[at + 6] = forces[index]![0]!
    into[at + 7] = forces[index]![1]!
  }
  link.open(broadcast)
  const draw: Draw = (at, receiver, sender) => draws.get(at)?.[receiver]?.[sender - Number(sender > receiver)] ?? 0
  for (; step <= last; step++) link.hear(broadcast, draw)
  return link
}

/**
 * Lists for each receiver of `link` the senders its table holds, in the order
 * of their indices.
 */
function held(link: Radio): number[][] {
  const slots = model.radio.slots
  return Array.from({ length: link.senders.length / slots }, (_, receiver) =>
    [...link.senders.subarray(receiver * slots, (receiver + 1) * slots)].filter(sender => sender >= 0)
      .toSorted((a, b) => a - b))
}

/**
 * Reads the slot of `receiver`'s table in `link` holding `sender`.
 */
const slotOf = (link: Radio, receiver: number, sender: number): number =>
  receiver * model.radio.slots
  + link.senders.subarray(receiver * model.radio.slots, (receiver + 1) * model.radio.slots).indexOf(sender)

describe('Radio.hear', () => {
  it.for(RUN.packets.map((step, packet) => [step, packet] as const))(
    'hears by the packet at step %i what the package’s radio hears from the same draws, within a thousandth',
    ([step, packet]) => {
      const link   = carried(RUN.flock, step, DRAWS)
      const tables = RUN.held[packet]!
      const apart  = tables.flatMap((senders, receiver) => senders.flatMap((sender, slot) =>
        RUN.positions[packet]![receiver]![slot]!.map((value, axis) =>
          Math.abs(link.inbox[slotOf(link, receiver, sender) * SENT + axis]! - value))))
      expect(held(link)).toEqual(tables)
      expect(Math.max(...apart)).toBeLessThan(1e-3)
    }
  )

  it.for(CROSSING.packets.map((step, packet) => [step, packet] as const))(
    'holds by the packet at step %i the senders the package’s radio holds as a flock crosses itself',
    ([step, packet]) => {
      expect(held(carried(CROSSING.flock, step))).toEqual(CROSSING.held[packet])
    }
  )
})
