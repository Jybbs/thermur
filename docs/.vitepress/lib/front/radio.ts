import type { Model } from './slots.ts'

import { Curve } from './curve.ts'

/**
 * Writes the broadcast of the drone at an index into an array from a place on.
 */
export type Broadcast = (index: number, into: Float64Array, at: number) => void

/**
 * The numbers each drone broadcasts, in the order a broadcast holds them: its
 * east, north, and height, its velocity's three parts, and its zig's countdown
 * and side.
 */
export const SENT = 8

/**
 * Measures the distance between the position at `at` of `from` and the one at
 * `to` of `into`, in meters.
 */
function apart(at: number, from: Float64Array, into: Float64Array, to: number): number {
  const dx = from[at]! - into[to]!, dy = from[at + 1]! - into[to + 1]!, dz = from[at + 2]! - into[to + 2]!
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/**
 * The package's radio under its default link, in which each drone broadcasts at
 * the link's rate, each packet arriving the link's delay late, only from a
 * sender within the link's range as it was sent, and only where the link does
 * not lose it, and each drone forgets a sender once the link's hold passes with
 * no packet from it, a drone never hearing itself. Each drone holds in its table
 * the link's `slots` nearest senders it hears, and learns a sender beyond the
 * table from the neighbor list each member of the table broadcast and from the
 * `slots` drones beside it along the Z-order curve of the broadcasts. What drone
 * `r` last heard of drone `s` sits at `(r * count + s) * SENT` of `inbox`, laid
 * out as `SENT` gives, and whether its table holds that sender at
 * `r * count + s` of `heard`. Every array a step reads and writes is built once,
 * so a step allocates nothing.
 */
export class Radio {
  readonly heard     : Uint8Array
  readonly inbox     : Float64Array
  /**
   * Whether the packet from each sender reaches each receiver at the packet
   * being delivered, by receiver and sender.
   */
  readonly #arrives  : Uint8Array
  readonly #count    : number
  readonly #curve    : Curve
  /**
   * The steps the link's delay spans.
   */
  readonly #delay    : number
  /**
   * The distances and drones of the nearest senders a receiver has found so
   * far, nearest first.
   */
  readonly #distance : Float64Array
  readonly #found    : Int32Array
  /**
   * The table each drone held at each of the last steps the link keeps, its
   * neighbor list, laid out as `#table` is under the step's place.
   */
  readonly #lists    : Int32Array
  readonly #model    : Model
  /**
   * The steps between two packets, at least one.
   */
  readonly #period   : number
  readonly #received : Int32Array
  /**
   * The receiver each sender was last weighed for, so a sender two routes find
   * is weighed once.
   */
  readonly #seen     : Int32Array
  readonly #sent     : Float64Array
  readonly #slots    : number
  /**
   * The senders each drone's table holds, `slots` to a drone and -1 for an
   * empty slot.
   */
  readonly #table    : Int32Array
  #filled            = 0
  #step              = 0

  constructor(count: number, model: Model) {
    const { plan, radio } = model
    const kept     = Math.round(radio.delay_s / plan.step_s) + 1
    this.#count    = count
    this.#model    = model
    this.#slots    = radio.slots
    this.#delay    = kept - 1
    this.#period   = Math.max(Math.round(1 / (radio.rate_hz * plan.step_s)), 1)
    this.heard     = new Uint8Array(count * count)
    this.inbox     = new Float64Array(count * count * SENT)
    this.#arrives  = new Uint8Array(count * count)
    this.#received = new Int32Array(count * count)
    this.#sent     = new Float64Array(kept * count * SENT)
    this.#table    = new Int32Array(count * radio.slots).fill(-1)
    this.#lists    = new Int32Array(kept * count * radio.slots).fill(-1)
    this.#curve    = new Curve(count)
    this.#seen     = new Int32Array(count).fill(-1)
    this.#distance = new Float64Array(radio.slots)
    this.#found    = new Int32Array(radio.slots)
  }

  /**
   * Finds whether the packet at `base` from each sender reaches each receiver,
   * from within the link's range of it as it was sent and where `random`,
   * drawn once for each receiver and each other drone in order, falls at or
   * past `loss`.
   */
  #arrive(base: number, loss: number, random: () => number): void {
    const count = this.#count
    const range = this.#model.radio.range_m
    for (let receiver = 0; receiver < count; receiver++) {
      for (let sender = 0; sender < count; sender++) {
        if (sender === receiver) continue
        const kept = random() >= loss
        this.#arrives[receiver * count + sender] = Number(
          kept && apart(base + sender * SENT, this.#sent, this.#sent, base + receiver * SENT) <= range
        )
      }
    }
  }

  /**
   * Weighs for `receiver` the senders its table holds that it still hears,
   * those the neighbor lists at `lists` of the members whose packet at `base`
   * reaches it name, and the `slots` drones beside it along the curve, then
   * keeps the nearest as its table.
   */
  #choose(base: number, lists: number, now: number, receiver: number): void {
    const count = this.#count
    const slots = this.#slots
    const row   = receiver * slots
    const hold  = this.#model.radio.hold_periods * this.#period
    this.#filled = 0
    for (let slot = row; slot < row + slots; slot++) {
      const member = this.#table[slot]!
      if (member < 0) continue
      const pair = receiver * count + member
      this.heard[pair]   = 0
      this.#seen[member] = receiver
      const alive        = this.#arrives[pair] || this.#step - this.#received[pair]! < hold
      if (alive) this.#weigh(base, member, now, receiver)
    }
    for (let slot = row; slot < row + slots; slot++) {
      const member = this.#table[slot]!
      if (member < 0 || !this.#arrives[receiver * count + member]) continue
      for (let at = lists + member * slots; at < lists + (member + 1) * slots; at++) {
        this.#offer(base, this.#lists[at]!, now, receiver)
      }
    }
    for (let offset = -(slots >> 1); offset < slots - (slots >> 1); offset++) {
      this.#offer(base, this.#curve.beside(receiver, offset), now, receiver)
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
   * `base` from each one whose packet reaches it.
   */
  #keep(base: number, receiver: number): void {
    const count = this.#count
    const row   = receiver * this.#slots
    this.#table.fill(-1, row, row + this.#slots)
    for (let rank = 0; rank < this.#filled; rank++) {
      const sender = this.#found[rank]!
      const pair   = receiver * count + sender
      if (this.#arrives[pair]) {
        this.inbox.set(this.#sent.subarray(base + sender * SENT, base + (sender + 1) * SENT), pair * SENT)
        this.#received[pair] = this.#step
      }
      this.heard[pair]        = 1
      this.#table[row + rank] = sender
    }
  }

  /**
   * Weighs `sender` for `receiver` where no route found it before and its
   * packet reaches the receiver.
   */
  #offer(base: number, sender: number, now: number, receiver: number): void {
    if (sender < 0 || this.#seen[sender] === receiver) return
    this.#seen[sender] = receiver
    if (this.#arrives[receiver * this.#count + sender]) this.#weigh(base, sender, now, receiver)
  }

  /**
   * Places `sender` at `distance` among the nearest senders found so far,
   * nearest first and a tie going to the lower index, where it is nearer than
   * the farthest of a full table.
   */
  #rank(distance: number, sender: number): void {
    const distances = this.#distance
    const found     = this.#found
    if (this.#filled === this.#slots && !this.#farther(distance, this.#filled - 1, sender)) return
    let rank = Math.min(this.#filled, this.#slots - 1)
    for (; rank > 0 && this.#farther(distance, rank - 1, sender); rank--) {
      distances[rank] = distances[rank - 1]!
      found[rank]     = found[rank - 1]!
    }
    distances[rank] = distance
    found[rank]     = sender
    this.#filled    = Math.min(this.#filled + 1, this.#slots)
  }

  /**
   * Ranks `sender` among the nearest senders `receiver` has found, by the
   * position its packet at `base` carries where one reaches the receiver and
   * otherwise by the position the receiver holds, from where the receiver
   * broadcasts at `now`.
   */
  #weigh(base: number, sender: number, now: number, receiver: number): void {
    const pair  = receiver * this.#count + sender
    const fresh = this.#arrives[pair]
    const at    = fresh ? base + sender * SENT : pair * SENT
    this.#rank(apart(at, fresh ? this.#sent : this.inbox, this.#sent, now + receiver * SENT), sender)
  }

  /**
   * Keeps every drone's broadcast `broadcast` writes at this step and its table
   * as its neighbor list, then, where a packet falls at this step, delivers the
   * broadcasts sent the link's delay earlier, each lost where `random` falls
   * under the link's loss, to each receiver from the senders its table holds,
   * from those the neighbor lists the members it hears sent with them name, and
   * from the `slots` drones beside it along the Z-order curve of those
   * broadcasts. Each receiver then keeps in its table the `slots` nearest of
   * these and of the senders it still holds, ranked by the positions it heard,
   * forgetting a sender no packet reached for the link's hold, which, as a
   * whole number of packet periods, runs out only where a packet falls.
   */
  hear(broadcast: Broadcast, random: () => number): void {
    const count = this.#count
    const kept  = this.#delay + 1
    const now   = this.#step % kept
    const late  = (((this.#step - this.#delay) % kept) + kept) % kept
    for (let index = 0; index < count; index++) broadcast(index, this.#sent, (now * count + index) * SENT)
    this.#lists.set(this.#table, now * this.#table.length)
    if (this.#step % this.#period === 0) {
      this.#arrive(late * count * SENT, this.#model.radio.loss, random)
      this.#curve.through(late * count * SENT, this.#sent, SENT)
      for (let receiver = 0; receiver < count; receiver++) {
        this.#choose(late * count * SENT, late * this.#table.length, now * count * SENT, receiver)
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
    this.heard.fill(0)
    this.#received.fill(0)
    this.#step = 0
    for (let receiver = 0; receiver < count; receiver++) {
      this.#filled = 0
      for (let sender = 0; sender < count; sender++) {
        const distance = apart(sender * SENT, sent, sent, receiver * SENT)
        const within   = sender !== receiver && distance <= range
        this.#arrives[receiver * count + sender] = Number(within)
        if (within) this.#rank(distance, sender)
      }
      this.#keep(0, receiver)
    }
    for (let at = 0; at < this.#lists.length; at += this.#table.length) this.#lists.set(this.#table, at)
  }
}
