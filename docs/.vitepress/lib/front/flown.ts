import * as common    from '../common.ts'
import source         from '../rendered/composite.json' with { type: 'json' }
import { Composite }  from './composite.ts'
import type { Draw }  from './radio.ts'
import type { Model } from './slots.ts'

/**
 * A flight of the package's composite from one seed, which draws every number
 * a step reads from that seed: the Brownian increments and the zig draws from
 * the Mulberry32 sequence `random` gives, and each packet's loss from the hash
 * of the seed, the step, and its pair. `random` also serves a drawing's own
 * draws taken before `launch`, so a flight from one seed flies alike every
 * time. Every array a step reads and writes is built once, so a step allocates
 * nothing.
 */
export class Flown {
  readonly composite   : Composite
  readonly random      : () => number
  readonly #draw       : Draw
  readonly #draws      : Float64Array
  readonly #increments : Float64Array
  readonly #normal     : () => number
  #steps               = 0

  constructor(count: number, seed: number, model: Model = source) {
    this.composite   = new Composite(count, model)
    this.random      = common.mulberry(seed)
    this.#normal     = common.gaussian(this.random)
    this.#draw       = (step, receiver, sender) => common.hashed(seed, step, receiver, sender)
    this.#draws      = new Float64Array(count)
    this.#increments = new Float64Array(count * 3)
  }

  /**
   * Reads the time the flight has flown, in seconds since its launch.
   */
  get seconds(): number {
    return this.#steps * this.composite.model.plan.step_s
  }

  /**
   * Flies the composite one step under fresh Brownian increments and zig draws,
   * with the accelerations `pushes` adds to the slots, laid out as the
   * composite's `positions` is.
   */
  advance(pushes: ArrayLike<number>): void {
    const root = Math.sqrt(this.composite.model.plan.step_s)
    for (let i = 0; i < this.#increments.length; i++) this.#increments[i] = this.#normal() * root
    for (let i = 0; i < this.#draws.length; i++) this.#draws[i] = this.random()
    this.composite.hear(this.#draw)
    this.composite.advance(this.#increments, this.#draws, pushes)
    this.#steps++
  }

  /**
   * Launches the flock as the plan launches one, from the seed's next draws.
   */
  launch(): void {
    this.composite.launch(this.#normal)
    this.#steps = 0
  }
}
