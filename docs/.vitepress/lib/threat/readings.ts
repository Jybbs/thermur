import * as common    from '../common.ts'
import type { Front } from './front.ts'

/**
 * Where a plot writes its reading: the time the rail shows, the rail laid over
 * the plot, and the value of each chip beneath it.
 */
export interface Readout {
  aim   : { value: number }
  first : HTMLElement
  front : HTMLElement
  last  : HTMLElement
  rail  : HTMLElement
  time  : HTMLElement
}

/**
 * Words the time after `seconds` for a screen reader, beside how far east the
 * front stands then.
 */
export const spoken = (front: Front, seconds: number): string =>
  `${seconds} seconds, the front ${Math.round(front.at(seconds))} meters east`

/**
 * Writes the reading of the plot into the page: the time, the front, and the
 * air at the first and the last cell into the chips, and the time a reader set
 * into the rail, each only as it changes and each chip's string built once.
 */
export class Readings {
  #first            = ''
  readonly #front   : Front
  readonly #fronts  : readonly string[]
  readonly #kelvins = new Map<number, string>()
  readonly #out     : Readout
  #second           = -1
  #target           = -1
  readonly #times   : readonly string[]

  constructor(front: Front, out: Readout) {
    this.#front   = front
    this.#out     = out
    const seconds = Array.from({ length: Math.round(front.span) + 1 }, (_, second) => second)
    const meters  = seconds.map(second => Math.round(front.at(second)))
    this.#times   = seconds.map(second => `${second} s`)
    this.#fronts  = meters.map(reach => `${reach} m`)
  }

  /**
   * Reads the air at `x` after `seconds` as the chips write it, to a tenth of a
   * kelvin.
   */
  #kelvin(seconds: number, x: number): string {
    const tenths = Math.round(this.#front.air(this.#front.lowest, seconds, x) * 10)
    return common.cached(this.#kelvins, tenths, () => `${(tenths / 10).toFixed(1)} K`)
  }

  /**
   * Writes the reading after `seconds`, with `target` the time a reader set.
   */
  write(seconds: number, target: number): void {
    const { aim, first, front, last, time } = this.#out
    const aimed  = Math.round(target)
    const second = Math.round(seconds)
    const air    = this.#kelvin(seconds, this.#front.center(0))
    if (aimed !== this.#target) {
      this.#target = aimed
      aim.value    = aimed
    }
    if (second !== this.#second) {
      this.#second      = second
      time.textContent  = this.#times[second]!
      front.textContent = this.#fronts[second]!
      last.textContent  = this.#kelvin(seconds, this.#front.far)
    }
    if (air !== this.#first) {
      this.#first       = air
      first.textContent = air
    }
  }
}
