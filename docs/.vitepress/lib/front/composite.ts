import { ranked }                      from '../common.ts'
import source                          from '../rendered/composite.json' with { type: 'json' }
import { launched }                    from './launch.ts'
import { type Broadcast, Radio, SENT } from './radio.ts'
import { type Model, type Place, TINY, capped, excited, home, zigging } from './slots.ts'

/**
 * The kind of component each slot holds that `Composite` copies, which a model
 * naming any other fails to build.
 */
const KINDS = {
  cohesion : 'Centered',
  contact  : 'HardSphere',
  coupling : 'Constant',
  heading  : 'Consensus',
  noise    : 'Zig',
  roost    : 'Home',
  speed    : 'Cruise'
}

/**
 * The package's composite under its default slots, stepped the way a flight
 * steps it. Each step the radio delivers, at its rate and late by its delay,
 * the broadcasts its link keeps from the drones within its range, then each
 * drone ranks the senders it heard by the positions it heard, sums the
 * centered cohesion, the hard-sphere contact, the consensus at the constant
 * coupling, the zig, the roost, and the cruise into the rate of its command,
 * and moves through one Euler step of the plan's `step_s`, its velocity
 * carried toward its command by the plant, its command held to the plant's
 * bounds, and its zig started, copied, or counted down. Each drone's east,
 * north, and height sit in meters at three places of `positions`, and its
 * velocity, command, and zig state likewise in `velocities`, `commands`, and
 * `forces`. Every array a step reads and writes is built once, so a step
 * allocates nothing.
 */
export class Composite {
  readonly commands   : Float64Array
  readonly count      : number
  readonly forces     : Float64Array
  readonly model      : Model
  readonly positions  : Float64Array
  /**
   * The senders each drone ranked at the last step, nearest first by the
   * positions it heard and `-1` past the last heard, the plan's `neighbors` of
   * them for each drone in turn.
   */
  readonly ranks      : Int32Array
  /**
   * The roost's disk and height the drones are held over, the package's own
   * until a drawing places it.
   */
  readonly roost      : Place
  readonly velocities : Float64Array
  readonly #before    : Float64Array
  readonly #best      : Float64Array
  readonly #broadcast : Broadcast
  readonly #radio     : Radio
  readonly #rates     : Float64Array
  readonly #sides     : Float64Array
  readonly #sums      : Float64Array
  readonly #turns     : Float64Array
  readonly #zigs      : Float64Array

  constructor(count: number, model: Model = source) {
    for (const [slot, kind] of Object.entries(KINDS)) {
      const held = model[slot as keyof typeof KINDS].kind
      if (held !== kind) throw new Error(`The ${slot} slot holds ${held}, where the copy flies ${kind}`)
    }
    const { neighbors } = model.plan
    this.count          = count
    this.model          = model
    this.roost          = {
      center_m : [model.roost.center_m[0]!, model.roost.center_m[1]!],
      height_m : model.roost.height_m,
      radius_m : model.roost.radius_m
    }
    this.positions  = new Float64Array(count * 3)
    this.velocities = new Float64Array(count * 3)
    this.commands   = new Float64Array(count * 3)
    this.forces     = new Float64Array(count * 3)
    this.#before    = new Float64Array(count * 3)
    this.#rates     = new Float64Array(count * 3)
    this.#turns     = new Float64Array(count * 3)
    this.#zigs      = new Float64Array(count)
    this.#sides     = new Float64Array(count)
    this.#sums      = new Float64Array(11)
    this.#best      = new Float64Array(neighbors)
    this.ranks      = new Int32Array(count * neighbors).fill(-1)
    this.#radio     = new Radio(count, model)
    this.#broadcast = (index, into, at): void => {
      const from = index * 3
      for (let axis = 0; axis < 3; axis++) {
        into[at + axis]     = this.positions[from + axis]!
        into[at + 3 + axis] = this.velocities[from + axis]!
      }
      into[at + 6] = this.forces[from]!
      into[at + 7] = this.forces[from + 1]!
    }
  }

  /**
   * Ranks the senders the drone at `index` heard by the positions it heard into
   * its place in `ranks`, their distances in `#best`, returning its centrality,
   * the length of the mean bearing to every sender it heard.
   */
  #rank(index: number): number {
    const { count, positions } = this
    const best  = this.#best
    const inbox = this.#radio.inbox
    const at    = index * 3
    const base  = index * best.length
    let cx = 0, cy = 0, cz = 0, held = 0
    best.fill(Infinity)
    this.ranks.fill(-1, base, base + best.length)
    for (let sender = 0; sender < count; sender++) {
      const pair = index * count + sender
      if (sender === index || !this.#radio.heard[pair]) continue
      const o        = pair * SENT
      const dx       = inbox[o]! - positions[at]!
      const dy       = inbox[o + 1]! - positions[at + 1]!
      const dz       = inbox[o + 2]! - positions[at + 2]!
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
      const apart    = Math.max(distance, TINY)
      cx += dx / apart
      cy += dy / apart
      cz += dz / apart
      held++
      ranked(best, distance, this.ranks, sender, base)
    }
    return Math.hypot(cx, cy, cz) / Math.max(held, 1)
  }

  /**
   * Sums over the neighbors `#rank` ranked for the drone at `index` into `#sums`
   * the differences of their velocities from its command, the bearings past
   * the cohesion's hard sphere outside its blind angle, and the contact's
   * pushes along their bearings, then the counts outside the blind angle and
   * of every neighbor off its position, and counts its zigging neighbors and
   * their sides into `#zigs` and `#sides`.
   */
  #gather(index: number): void {
    const { cohesion, contact, noise }   = this.model
    const { commands, count, positions } = this
    const inbox                          = this.#radio.inbox
    const sums                           = this.#sums
    const at                             = index * 3
    const base                           = index * this.#best.length
    const cx                             = commands[at]!
    const cy                             = commands[at + 1]!
    const cz                             = commands[at + 2]!
    const blind                          = -Math.cos(cohesion.blind_deg * Math.PI / 360)
    const length                         = Math.max(Math.hypot(cx, cy, cz), TINY)
    sums.fill(0)
    this.#zigs[index]  = 0
    this.#sides[index] = 0
    for (let k = 0; k < this.#best.length; k++) {
      const sender = this.ranks[base + k]!
      if (sender < 0) break
      const o        = (index * count + sender) * SENT
      const distance = this.#best[k]!
      const dx       = inbox[o]! - positions[at]!
      const dy       = inbox[o + 1]! - positions[at + 1]!
      const dz       = inbox[o + 2]! - positions[at + 2]!
      for (let axis = 0; axis < 3; axis++) {
        sums[axis] = sums[axis]! + inbox[o + 3 + axis]! - commands[at + axis]!
      }
      if (distance <= 0) continue
      const forward  = dx * cx + dy * cy + dz * cz > blind * distance * length
      const pull     = forward && distance > cohesion.sphere_m ? 1 / distance : 0
      const strength = Math.exp(-((Math.max(distance - contact.sphere_m, 0) / contact.spread_m) ** 2) / 2)
                     / distance
      for (let axis = 0; axis < 3; axis++) {
        const offset   = inbox[o + axis]! - positions[at + axis]!
        sums[3 + axis] = sums[3 + axis]! + pull * offset
        sums[6 + axis] = sums[6 + axis]! + strength * offset
      }
      sums[9]  = sums[9]! + Number(forward)
      sums[10] = sums[10]! + 1
      if (zigging(noise, inbox[o + 6]!)) {
        this.#zigs[index]  = this.#zigs[index]! + 1
        this.#sides[index] = this.#sides[index]! + inbox[o + 7]!
      }
    }
  }

  /**
   * Sums the slots for the drone at `index` and the push `pushes` holds for it
   * into `#rates`, keeping the roost's part in `#turns`.
   */
  #sense(index: number, pushes: ArrayLike<number>): void {
    const { cohesion, contact, coupling, heading, noise, plan, speed } = this.model
    const { commands, forces, positions }                              = this
    const at         = index * 3
    const centrality = this.#rank(index)
    const sums       = this.#sums
    this.#gather(index)
    home(at, commands, this.#turns, this.roost, positions, this.model.roost)
    const vx        = commands[at]!, vy = commands[at + 1]!
    const pace      = Math.hypot(vx, vy, commands[at + 2]!)
    const countdown = forces[at]!
    const zig       = zigging(noise, countdown)
      ? forces[at + 1]! * noise.accel_m_s2 * (noise.refractory_s + noise.back_s < countdown ? 1 : -1)
        / Math.max(Math.hypot(vx, vy), TINY)
      : 0
    const together  = centrality * cohesion.weight_m_s2 / Math.max(sums[9]!, 1)
    const away      = -contact.weight_m_s2 / Math.max(sums[10]!, 1)
    const aligned   = coupling.coupling_per_s * (1 + heading.centering / plan.neighbors)
    const cruise    = (speed.cruise_m_s - pace) / Math.max(pace, TINY) / speed.relaxation_s
    for (let axis = 0; axis < 3; axis++) {
      this.#rates[at + axis] = together * sums[3 + axis]! + away * sums[6 + axis]! + aligned * sums[axis]!
                             + cruise * commands[at + axis]! + this.#turns[at + axis]! + pushes[at + axis]!
    }
    this.#rates[at]     = this.#rates[at]! - zig * vy
    this.#rates[at + 1] = this.#rates[at + 1]! + zig * vx
  }

  /**
   * Moves every drone through one step under the Brownian `increments` on each
   * command, the uniform `draws` each drone's zig reads, and the accelerations
   * `pushes` adds to the slots, `increments` and `pushes` laid out as
   * `positions` is.
   */
  advance(increments: ArrayLike<number>, draws: ArrayLike<number>, pushes: ArrayLike<number>): void {
    const { noise, plan, plant }              = this.model
    const { commands, positions, velocities } = this
    const rates                               = this.#rates
    const step                                = plan.step_s
    for (let index = 0; index < this.count; index++) this.#sense(index, pushes)
    this.#before.set(commands)
    for (let at = 0; at < positions.length; at += 3) {
      capped(this.#before, at, plant.climb_m_s, plant.level_m_s, -plant.descent_m_s)
      for (let axis = at; axis < at + 3; axis++) {
        this.#before[axis] = (this.#before[axis]! - velocities[axis]!) / plant.lag_s
      }
      capped(this.#before, at, plant.climbing_m_s2, plant.level_m_s2, -plant.descending_m_s2)
      for (let axis = at; axis < at + 3; axis++) {
        positions[axis]    = positions[axis]! + step * velocities[axis]!
        velocities[axis]   = velocities[axis]! + step * this.#before[axis]!
        this.#before[axis] = commands[axis]!
        commands[axis]     = commands[axis]! + step * rates[axis]! + noise.amplitude * increments[axis]!
      }
      capped(commands, at, plant.climb_m_s, plant.level_m_s, -plant.descent_m_s)
    }
    for (let index = 0; index < this.count; index++) {
      excited(
        this.forces,
        index * 3,
        this.#before,
        draws[index]!,
        this.model,
        this.#sides[index]!,
        this.#turns,
        this.#zigs[index]!
      )
    }
  }

  /**
   * Carries every drone's broadcast over the radio for this step, each packet
   * lost where `random` falls under the link's loss.
   */
  hear(random: () => number): void {
    this.#radio.hear(this.#broadcast, random)
  }

  /**
   * Launches the drones as the plan launches a flock, every draw from
   * `normal`, each commanding the velocity it flies at with no zig under way,
   * then opens the radio.
   */
  launch(normal: () => number): void {
    launched(normal, this.model.plan, this.positions, this.velocities)
    this.commands.set(this.velocities)
    this.forces.fill(0)
    this.open()
  }

  /**
   * Opens the radio on the flock as it stands, every drone having heard each
   * other drone within the radio's range.
   */
  open(): void {
    this.#radio.open(this.#broadcast)
  }
}
