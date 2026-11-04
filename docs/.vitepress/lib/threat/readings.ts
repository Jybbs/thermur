import * as common    from '../common.ts'
import type { Front } from './front.ts'

/**
 * The page's elements a plot writes its reading into: the rail a reader moves
 * and the value of each chip beneath it.
 */
export interface Readout {
  first : HTMLElement
  front : HTMLElement
  last  : HTMLElement
  rail  : HTMLInputElement
  time  : HTMLElement
}

/**
 * Writes the reading of the plot into the page: the time, the front, and the
 * air at the first and the last cell into the chips, and the time a reader set
 * into the rail, each only as it changes and each string built once.
 */
export class Readings {
  #first            = ''
  readonly #front   : Front
  readonly #fronts  : readonly string[]
  readonly #kelvins = new Map<number, string>()
  readonly #out     : Readout
  #second           = -1
  readonly #spoken  : readonly string[]
  #target           = -1
  readonly #times   : readonly string[]
  readonly #values  : readonly string[]

  constructor(front: Front, out: Readout) {
    this.#front   = front
    this.#out     = out
    const seconds = Array.from({ length: Math.round(front.span) + 1 }, (_, second) => second)
    const meters  = seconds.map(second => Math.round(front.at(second)))
    this.#times   = seconds.map(second => `${second} s`)
    this.#values  = seconds.map(String)
    this.#fronts  = meters.map(reach => `${reach} m`)
    this.#spoken  = seconds.map(second => `${second} seconds, the front ${meters[second]} meters east`)
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
    const { first, front, last, rail, time } = this.#out
    const aim    = Math.round(target)
    const second = Math.round(seconds)
    const air    = this.#kelvin(seconds, this.#front.center(0))
    if (aim !== this.#target) {
      this.#target = aim
      rail.value   = this.#values[aim]!
      rail.setAttribute('aria-valuetext', this.#spoken[aim]!)
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
