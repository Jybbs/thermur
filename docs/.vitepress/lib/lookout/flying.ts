import { FLIGHT }       from '../front/flock.ts'
import { Flown }        from '../front/flown.ts'
import type { Playing } from './flight.ts'

/**
 * A flight of the package's composite flown in the page from the plan's seed,
 * under no push beside the slots, moving on by whole steps of the plan's
 * `step_s` and no more than the site's flocks fly a frame.
 */
export class Flying implements Playing {
  readonly count      : number
  readonly positions  : Float64Array
  readonly velocities : Float64Array
  #clock              = 0
  readonly #flown     : Flown
  readonly #still     : Float64Array

  constructor(count: number, seed: number) {
    this.count      = count
    this.#flown     = new Flown(count, seed)
    this.#still     = new Float64Array(count * 3)
    this.positions  = this.#flown.composite.positions
    this.velocities = this.#flown.composite.velocities
    this.#flown.launch()
  }

  get seconds(): number {
    return this.#flown.seconds
  }

  play(dt: number): void {
    const step   = this.#flown.composite.model.plan.step_s
    this.#clock += Math.min(dt, FLIGHT.stretch)
    for (; this.#clock >= step; this.#clock -= step) this.#flown.advance(this.#still)
  }
}
