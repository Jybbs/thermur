import type { Model } from './slots.ts'

/**
 * Writes the broadcast of the drone at an index into an array from a place on.
 */
export type Broadcast = (index: number, into: Float64Array, at: number) => void

/**
 * Draws a number no loss falls under, so the link opens with every packet kept.
 */
const KEPT = (): number => 1

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
 * no packet from it, a drone never hearing itself. What drone `r` last heard of
 * drone `s` sits at `(r * count + s) * SENT` of `inbox`, laid out as `SENT`
 * gives, and whether it still hears that sender at `r * count + s` of `heard`.
 * Every array a step reads and writes is built once, so a step allocates
 * nothing.
 */
export class Radio {
  readonly heard     : Uint8Array
  readonly inbox     : Float64Array
  readonly #count    : number
  /**
   * The steps the link's delay spans.
   */
  readonly #delay    : number
  readonly #model    : Model
  /**
   * The steps between two packets, at least one.
   */
  readonly #period   : number
  readonly #received : Int32Array
  readonly #sent     : Float64Array
  #step              = 0

  constructor(count: number, model: Model) {
    const { plan, radio } = model
    this.#count    = count
    this.#model    = model
    this.#delay    = Math.round(radio.delay_s / plan.step_s)
    this.#period   = Math.max(Math.round(1 / (radio.rate_hz * plan.step_s)), 1)
    this.heard     = new Uint8Array(count * count)
    this.#received = new Int32Array(count * count)
    this.inbox     = new Float64Array(count * count * SENT)
    this.#sent     = new Float64Array((this.#delay + 1) * count * SENT)
  }

  /**
   * Delivers to every receiver the broadcast at `base` of the kept broadcasts
   * from each sender within the link's range of it, each delivery lost where
   * `random` falls under `loss`.
   */
  #deliver(base: number, random: () => number, loss: number): void {
    const count = this.#count
    const sent  = this.#sent
    const range = this.#model.radio.range_m ** 2
    const hold  = this.#model.radio.hold_periods * this.#period
    for (let receiver = 0; receiver < count; receiver++) {
      const r = base + receiver * SENT
      for (let sender = 0; sender < count; sender++) {
        if (sender === receiver) continue
        const pair = receiver * count + sender
        const s    = base + sender * SENT
        const dx   = sent[s]! - sent[r]!, dy = sent[s + 1]! - sent[r + 1]!, dz = sent[s + 2]! - sent[r + 2]!
        if (dx * dx + dy * dy + dz * dz <= range && random() >= loss) {
          for (let k = 0; k < SENT; k++) this.inbox[pair * SENT + k] = sent[s + k]!
          this.heard[pair]     = 1
          this.#received[pair] = this.#step
        } else if (this.#step - this.#received[pair]! >= hold) {
          this.heard[pair] = 0
        }
      }
    }
  }

  /**
   * Keeps every drone's broadcast `broadcast` writes at this step, then, where
   * a packet falls at this step, delivers the broadcasts sent the link's delay
   * earlier, each lost where `random` falls under the link's loss, and lets
   * each drone forget a sender no packet reached for the link's hold, which, as
   * a whole number of packet periods, runs out only where a packet falls.
   */
  hear(broadcast: Broadcast, random: () => number): void {
    const count = this.#count
    const kept  = this.#delay + 1
    const late  = this.#step - this.#delay
    for (let index = 0; index < count; index++) {
      broadcast(index, this.#sent, ((this.#step % kept) * count + index) * SENT)
    }
    if (this.#step % this.#period === 0) {
      this.#deliver((((late % kept) + kept) % kept) * count * SENT, random, this.#model.radio.loss)
    }
    this.#step++
  }

  /**
   * Opens the link on the broadcasts `broadcast` writes, every drone having
   * heard each other drone within the link's range, and every broadcast the
   * link keeps being this one.
   */
  open(broadcast: Broadcast): void {
    const block = this.#count * SENT
    for (let index = 0; index < this.#count; index++) broadcast(index, this.#sent, index * SENT)
    for (let at = block; at < this.#sent.length; at += block) this.#sent.copyWithin(at, 0, block)
    this.heard.fill(0)
    this.#received.fill(0)
    this.#step = 0
    this.#deliver(0, KEPT, 0)
  }
}
