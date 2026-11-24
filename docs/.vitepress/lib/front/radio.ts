import type { Model } from './slots.ts'

import { apart } from '../common.ts'
import { Curve } from './curve.ts'

/**
 * Writes the broadcast of the drone at an index into an array from a place on.
 */
export type Broadcast = (index: number, into: Float64Array, at: number) => void

/**
 * Draws the uniform number the packet from a sender to a receiver at a step
 * compares with the link's loss, given the step, the receiver, and the sender,
 * as the package's `Radio.draws` does, so one pair at one step draws the same
 * number wherever a search finds it.
 */
export type Draw = (step: number, receiver: number, sender: number) => number

/**
 * The numbers each drone broadcasts, in the order a broadcast holds them: its
 * east, north, and height, its velocity's three parts, and its zig's countdown
 * and side.
 */
export const SENT = 8

/**
 * The package's radio under its default link, in which each drone broadcasts at
 * the link's rate, each packet arriving the link's delay late, only from a
 * sender within the link's range as it was sent, and only where the link does
 * not lose it, and each drone forgets a sender once the link's hold passes with
 * no packet from it, a drone never hearing itself. Each drone holds in its table
 * the link's `slots` nearest senders it hears, and learns a sender beyond the
 * table from the neighbor list each member of the table broadcast and from each
 * other drone in the `slots` places along the Z-order curve of the broadcasts
 * that start `slots / 2` before its own. The sender in slot `k` of drone `r`'s
 * table sits at `r * slots + k` of `senders`, `-1` for an empty slot, and what
 * `r` last heard of it at `(r * slots + k) * SENT` of `inbox`, laid out as
 * `SENT` gives, as the package's `Inbox` lays its slots out. Every array a step
 * reads and writes is built once, growing with the flock's size alone, so a
 * step allocates nothing.
 */
export class Radio {
  readonly inbox     : Float64Array
  readonly senders   : Int32Array
  readonly #count    : number
  readonly #curve    : Curve
  /**
   * The steps the link's delay spans.
   */
  readonly #delay    : number
  /**
   * The distances, drones, and slots of the nearest senders a receiver has
   * found so far, nearest first, the slot being the one of the receiver's table
   * a sender heard through the hold alone sat in, and `-1` for a sender whose
   * packet reaches it.
   */
  readonly #distance : Float64Array
  #filled            = 0
  readonly #found    : Int32Array
  readonly #from     : Int32Array
  /**
   * The broadcasts the receiver's table held as its choice began, laid out as
   * one row of `inbox`.
   */
  readonly #heard    : Float64Array
  /**
   * The table each drone held at each of the last steps the link keeps, its
   * neighbor list, laid out as `senders` is under the step's place.
   */
  readonly #lists    : Int32Array
  readonly #model    : Model
  /**
   * The steps between two packets, at least one.
   */
  readonly #period   : number
  /**
   * Whether the packet from each member of the receiver's table reaches it at
   * the packet being delivered, by slot.
   */
  readonly #reached  : Uint8Array
  /**
   * The step at which each drone last took a packet from the sender in each
   * slot, laid out as `senders` is.
   */
  readonly #received : Int32Array
  /**
   * The receiver each sender was last weighed for, so a sender two routes find
   * is weighed once.
   */
  readonly #seen     : Int32Array
  readonly #sent     : Float64Array
  readonly #slots    : number
  #step              = 0
  /**
   * The steps the receiver's table took each broadcast at as its choice began,
   * by slot.
   */
  readonly #taken    : Int32Array

  constructor(count: number, model: Model) {
    const { plan, radio } = model
    const kept     = Math.round(radio.delay_s / plan.step_s) + 1
    this.#count    = count
    this.#model    = model
    this.#slots    = radio.slots
    this.#delay    = kept - 1
    this.#period   = Math.max(Math.round(1 / (radio.rate_hz * plan.step_s)), 1)
    this.inbox     = new Float64Array(count * radio.slots * SENT)
    this.senders   = new Int32Array(count * radio.slots).fill(-1)
    this.#received = new Int32Array(count * radio.slots)
    this.#sent     = new Float64Array(kept * count * SENT)
    this.#lists    = new Int32Array(kept * count * radio.slots).fill(-1)
    this.#curve    = new Curve(count)
    this.#seen     = new Int32Array(count).fill(-1)
    this.#distance = new Float64Array(radio.slots)
    this.#found    = new Int32Array(radio.slots)
    this.#from     = new Int32Array(radio.slots)
    this.#heard    = new Float64Array(radio.slots * SENT)
    this.#taken    = new Int32Array(radio.slots)
    this.#reached  = new Uint8Array(radio.slots)
  }

  /**
   * Finds whether the packet at `base` from `sender` reaches `receiver`, from
   * within the link's range of it as it was sent and where the pair's `draw`
   * falls under the link's chance of keeping it.
   */
  #arrives(base: number, draw: Draw, receiver: number, sender: number): boolean {
    const { loss, range_m } = this.#model.radio
    return draw(this.#step, receiver, sender) < 1 - loss
        && apart(base + sender * SENT, this.#sent, this.#sent, base + receiver * SENT) <= range_m
  }

  /**
   * Weighs for `receiver` the senders its table holds that it still hears,
   * those the neighbor lists at `lists` of the members whose packet at `base`
   * reaches it name, and each other drone in the `slots` places along the
   * curve that start `slots / 2` before its own, then keeps the nearest as its
   * table.
   */
  #choose(base: number, draw: Draw, lists: number, now: number, receiver: number): void {
    const slots = this.#slots
    const row   = receiver * slots
    const hold  = this.#model.radio.hold_periods * this.#period
    this.#filled = 0
    for (let slot = 0; slot < slots; slot++) {
      const member        = this.senders[row + slot]!
      this.#reached[slot] = 0
      if (member < 0) continue
      this.#seen[member]  = receiver
      this.#reached[slot] = Number(this.#arrives(base, draw, receiver, member))
      if (this.#reached[slot]) this.#weigh(base, -1, member, now, receiver)
      else if (this.#step - this.#received[row + slot]! < hold) this.#weigh(base, slot, member, now, receiver)
    }
    for (let slot = 0; slot < slots; slot++) {
      if (!this.#reached[slot]) continue
      const member = this.senders[row + slot]!
      for (let at = lists + member * slots; at < lists + (member + 1) * slots; at++) {
        this.#offer(base, draw, this.#lists[at]!, now, receiver)
      }
    }
    for (let offset = -(slots >> 1); offset < slots - (slots >> 1); offset++) {
      this.#offer(base, draw, this.#curve.beside(receiver, offset), now, receiver)
    }
    this.#keep(base, receiver)
  }

  /**
   * Finds whether the sender found at `rank` sits farther than `sender` at
   * `distance`, a tie going to the lower index.
   */
  #farther(distance: number, rank: number, sender: number): boolean {
    const held = this.#distance[rank]!
    return held > distance || held === distance && this.#found[rank]! > sender
  }

  /**
   * Writes the senders found for `receiver` as its table, taking the packet at
   * `base` from each one whose packet reaches it and carrying over what its
   * table held of each other.
   */
  #keep(base: number, receiver: number): void {
    const slots = this.#slots
    const row   = receiver * slots
    for (let k = 0; k < slots * SENT; k++) this.#heard[k] = this.inbox[row * SENT + k]!
    for (let slot = 0; slot < slots; slot++) this.#taken[slot] = this.#received[row + slot]!
    this.senders.fill(-1, row, row + slots)
    for (let rank = 0; rank < this.#filled; rank++) {
      const from   = this.#from[rank]!
      const sender = this.#found[rank]!
      const at     = from < 0 ? base + sender * SENT : from * SENT
      const held   = from < 0 ? this.#sent : this.#heard
      for (let k = 0; k < SENT; k++) this.inbox[(row + rank) * SENT + k] = held[at + k]!
      this.#received[row + rank] = from < 0 ? this.#step : this.#taken[from]!
      this.senders[row + rank]   = sender
    }
  }

  /**
   * Weighs `sender` for `receiver` where no route found it before and its
   * packet reaches the receiver under `draw`.
   */
  #offer(base: number, draw: Draw, sender: number, now: number, receiver: number): void {
    if (sender < 0 || sender === receiver || this.#seen[sender] === receiver) return
    this.#seen[sender] = receiver
    if (this.#arrives(base, draw, receiver, sender)) this.#weigh(base, -1, sender, now, receiver)
  }

  /**
   * Places `sender` at `distance`, heard through the slot `from`, among the
   * nearest senders found so far, nearest first and a tie going to the lower
   * index, where it is nearer than the farthest of a full table.
   */
  #rank(distance: number, from: number, sender: number): void {
    const distances = this.#distance
    const found     = this.#found
    if (this.#filled === this.#slots && !this.#farther(distance, this.#filled - 1, sender)) return
    let rank = Math.min(this.#filled, this.#slots - 1)
    for (; rank > 0 && this.#farther(distance, rank - 1, sender); rank--) {
      distances[rank]  = distances[rank - 1]!
      found[rank]      = found[rank - 1]!
      this.#from[rank] = this.#from[rank - 1]!
    }
    distances[rank]  = distance
    found[rank]      = sender
    this.#from[rank] = from
    this.#filled     = Math.min(this.#filled + 1, this.#slots)
  }

  /**
   * Ranks `sender` among the nearest senders `receiver` has found, by the
   * position its packet at `base` carries where `from` is `-1` and otherwise by
   * the position the slot `from` of the receiver's table holds, from where the
   * receiver broadcasts at `now`.
   */
  #weigh(base: number, from: number, sender: number, now: number, receiver: number): void {
    const at   = from < 0 ? base + sender * SENT : (receiver * this.#slots + from) * SENT
    const held = from < 0 ? this.#sent : this.inbox
    this.#rank(apart(at, held, this.#sent, now + receiver * SENT), from, sender)
  }

  /**
   * Keeps every drone's broadcast `broadcast` writes at this step and its table
   * as its neighbor list, then, where a packet falls at this step, delivers the
   * broadcasts sent the link's delay earlier, each lost where `draw` for its
   * pair falls at or past the link's chance of keeping it, to each receiver from
   * the senders its table holds, from those the neighbor lists the members it
   * hears sent with them name, and from each other drone in the `slots` places
   * along the Z-order curve of those broadcasts that start `slots / 2` before
   * its own. Each receiver then keeps in its table the `slots` nearest of these
   * and of the senders it still holds, ranked by the positions it heard,
   * forgetting a sender no packet reached for the link's hold, which, as a
   * whole number of packet periods, runs out only where a packet falls.
   */
  hear(broadcast: Broadcast, draw: Draw): void {
    const count = this.#count
    const kept  = this.#delay + 1
    const now   = this.#step % kept
    const late  = (((this.#step - this.#delay) % kept) + kept) % kept
    for (let index = 0; index < count; index++) broadcast(index, this.#sent, (now * count + index) * SENT)
    this.#lists.set(this.senders, now * this.senders.length)
    if (this.#step % this.#period === 0) {
      this.#curve.through(late * count * SENT, this.#sent, SENT)
      for (let receiver = 0; receiver < count; receiver++) {
        this.#choose(late * count * SENT, draw, late * this.senders.length, now * count * SENT, receiver)
      }
    }
    this.#step++
  }

  /**
   * Opens the link on the broadcasts `broadcast` writes, every drone holding in
   * its table the `slots` nearest other drones within the link's range, and
   * every broadcast and neighbor list the link keeps being this one.
   */
  open(broadcast: Broadcast): void {
    const count = this.#count
    const block = count * SENT
    const range = this.#model.radio.range_m
    const sent  = this.#sent
    for (let index = 0; index < count; index++) broadcast(index, sent, index * SENT)
    for (let at = block; at < sent.length; at += block) sent.copyWithin(at, 0, block)
    this.#step = 0
    for (let receiver = 0; receiver < count; receiver++) {
      this.#filled = 0
      for (let sender = 0; sender < count; sender++) {
        const distance = apart(sender * SENT, sent, sent, receiver * SENT)
        if (sender !== receiver && distance <= range) this.#rank(distance, -1, sender)
      }
      this.#keep(0, receiver)
    }
    for (let at = 0; at < this.#lists.length; at += this.senders.length) this.#lists.set(this.senders, at)
  }
}
