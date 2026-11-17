import { describe, expect, it } from 'vitest'

import * as common    from '../../../lib/common.ts'
import { Composite }  from '../../../lib/front/composite.ts'
import type { Model } from '../../../lib/front/slots.ts'
import model          from '../../../lib/rendered/composite.json' with { type: 'json' }
import parity         from '../../../lib/rendered/parity.json' with { type: 'json' }

/**
 * The flocks `parity` holds, each beside the step the package moves it through.
 */
type Parity = typeof parity.flocks.rim

/**
 * The flock `parity` holds astride the roost's rim.
 */
const RIM = parity.flocks.rim

/**
 * No increment and no push on any drone of `parity`'s flocks.
 */
const STILL = new Float64Array(RIM.flock.positions.length * 3)

/**
 * Builds a copy of `held`'s flock, the rim's unless given, under a radio of
 * its delay, every drone having heard every other.
 */
function launched(held: Parity = RIM): Composite {
  const { commands, forces, positions, velocities } = held.flock
  const copy = new Composite(positions.length, { ...model, radio: { ...model.radio, delay_s: held.delay_s } })
  copy.positions.set(positions.flat())
  copy.velocities.set(velocities.flat())
  copy.commands.set(commands.flat())
  copy.forces.set(forces.flat())
  copy.open()
  return copy
}

/**
 * Reads the largest gap between a value `held` holds and the package's value at
 * the same place of `stepped`.
 */
const divergence = (held: Float64Array, stepped: number[][]): number =>
  Math.max(...stepped.flat().map((value, index) => Math.abs(held[index]! - value)))

/**
 * Ranks the other drones of `positions` by their distance from the first,
 * nearest first, the plan's `neighbors` of them.
 */
function nearest(positions: Float64Array): number[] {
  const best = new Float64Array(model.plan.neighbors).fill(Infinity)
  const into = new Int32Array(model.plan.neighbors).fill(-1)
  for (let other = 1; other < positions.length / 3; other++) {
    const offsets = [0, 1, 2].map(axis => positions[other * 3 + axis]! - positions[axis]!)
    common.ranked(best, Math.hypot(...offsets), into, other)
  }
  return [...into]
}

/**
 * Builds a copy of the rim's flock under a roost whose disk reaches far past
 * every drone.
 */
function inside(): Composite {
  const copy = launched()
  copy.roost.radius_m = 1e6
  return copy
}

describe('Composite.advance', () => {
  it.for(
    (['late', 'rim', 'roost'] as const).flatMap(flock =>
      (['positions', 'velocities', 'commands', 'forces'] as const).map(name => [flock, name] as const)
    )
  )('steps the %s flock’s %s from one state to the package’s, within a thousandth', ([flock, name]) => {
    const held = parity.flocks[flock]
    const copy = launched(held)
    copy.advance(held.increments.flat(), held.draws, STILL)
    expect(divergence(copy[name], held.stepped[name])).toBeLessThan(1e-3)
  })

  it('holds every command inside the plant’s horizontal and vertical bounds', () => {
    const copy = launched()
    const fast = new Float64Array(copy.positions.length)
    for (let at = 0; at < fast.length; at += 3) {
      const pace = Math.hypot(copy.commands[at]!, copy.commands[at + 1]!, copy.commands[at + 2]!)
      for (let axis = at; axis < at + 3; axis++) fast[axis] = 5000 * copy.commands[axis]! / pace
      fast[at + 2] = fast[at + 2]! + 5000
    }
    copy.advance(STILL, new Float64Array(copy.count).fill(1), fast)
    for (let at = 0; at < copy.commands.length; at += 3) {
      expect(Math.hypot(copy.commands[at]!, copy.commands[at + 1]!)).toBeCloseTo(model.plant.level_m_s, 9)
      expect(copy.commands[at + 2]).toBeCloseTo(model.plant.climb_m_s, 9)
    }
  })

  it('starts a resting drone’s zig off a zigging neighbor where its draw is zero, and never where it is one', () => {
    const resting = RIM.flock.forces.findIndex(([countdown]) => countdown === 0)
    const off     = inside()
    const on      = inside()
    off.advance(STILL, new Float64Array(off.count).fill(1), STILL)
    on.advance(STILL, new Float64Array(on.count), STILL)
    expect(off.forces[resting * 3]).toBe(0)
    expect(on.forces[resting * 3]).toBeGreaterThan(model.noise.refractory_s)
  })

  it('keeps every command finite where two drones sit on one point', () => {
    const copy = launched()
    copy.positions.copyWithin(3, 0, 3)
    copy.open()
    copy.advance(RIM.increments.flat(), RIM.draws, STILL)
    expect(copy.commands.every(Number.isFinite)).toBe(true)
  })
})

describe('Composite.ranks', () => {
  it('ranks each drone’s senders by where its radio heard them rather than where they now fly', () => {
    const copy  = launched()
    const heard = nearest(copy.positions)
    const moved = copy.positions.slice(3)
    for (let other = 1; other < copy.count; other++) {
      copy.positions.set(moved.subarray((copy.count - 1 - other) * 3, (copy.count - other) * 3), other * 3)
    }
    expect(nearest(copy.positions)).not.toEqual(heard)
    copy.advance(STILL, new Float64Array(copy.count).fill(1), STILL)
    expect([...copy.ranks.subarray(0, model.plan.neighbors)]).toEqual(heard)
  })
})

describe('Composite.hear', () => {
  it('forgets every sender once the radio’s hold passes with every packet lost, flying as a drone alone', () => {
    const copy  = launched()
    const calm  = new Float64Array(copy.count).fill(1)
    const hold  = model.radio.hold_periods * Math.round(1 / (model.radio.rate_hz * model.plan.step_s))
    for (let step = 0; step <= hold; step++) {
      copy.hear(() => 0)
      copy.advance(STILL, calm, STILL)
    }
    copy.hear(() => 0)
    const alone = new Composite(1)
    const at    = 5 * 3
    for (const name of ['positions', 'velocities', 'commands', 'forces'] as const) {
      alone[name].set(copy[name].subarray(at, at + 3))
    }
    alone.open()
    copy.advance(STILL, calm, STILL)
    alone.advance(new Float64Array(3), [1], new Float64Array(3))
    expect([...copy.commands.subarray(at, at + 3)]).toEqual([...alone.commands])
  })
})

describe('Composite.launch', () => {
  it('launches each drone flying and commanding the cruise speed within the scatter of the course', () => {
    const copy = new Composite(200)
    copy.launch(common.gaussian(common.mulberry(3)))
    const course = model.plan.course_deg * Math.PI / 180
    for (let at = 0; at < copy.velocities.length; at += 3) {
      const [east, north] = [copy.velocities[at]!, copy.velocities[at + 1]!]
      expect(Math.hypot(...copy.velocities.subarray(at, at + 3))).toBeCloseTo(model.plan.cruise_m_s, 6)
      expect(Math.cos(course) * east + Math.sin(course) * north).toBeGreaterThan(model.plan.cruise_m_s * 0.8)
    }
    expect([...copy.commands]).toEqual([...copy.velocities])
  })

  it('launches a cloud longer along the course than across it and thinnest in height', () => {
    const copy   = new Composite(400)
    copy.launch(common.gaussian(common.mulberry(5)))
    const spread = (axis: number): number => {
      const values = copy.positions.filter((_, index) => index % 3 === axis)
      const mean   = values.reduce((sum, value) => sum + value, 0) / values.length
      return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
    }
    expect(spread(0)).toBeGreaterThan(spread(1))
    expect(spread(1)).toBeGreaterThan(spread(2))
  })
})

describe('Composite', () => {
  it.for(['cohesion', 'contact', 'coupling', 'heading', 'noise', 'roost', 'speed'] as const)(
    'refuses a model whose %s slot holds a component the copy does not fly',
    slot => {
      const other = { ...model, [slot]: { ...model[slot], kind: 'Other' } } as Model
      expect(() => new Composite(4, other)).toThrow(`The ${slot} slot holds Other`)
    }
  )
})
