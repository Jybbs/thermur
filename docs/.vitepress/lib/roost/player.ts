import { still } from '@thermur/kit'

import * as common                   from '../common.ts'
import { FRAMES, type Flight, PLAN } from './flight.ts'

/**
 * One stretch of the loop: its length in seconds of the page's own time, and
 * for each share of it run, the moment of the flight shown, how strongly the
 * drawing shows, and how far the camera is lifted toward the whole roost, each
 * of the last two from 0 to 1.
 */
interface Stretch {
  lift? : (u: number) => number
  span  : number
  t     : (u: number) => number
  veil? : (u: number) => number
}

/**
 * The seconds of the page's own time the flight takes at its pace.
 */
const FLOWN = PLAN.seconds / PLAN.pace

/**
 * The seconds of the page's own time the camera takes to pan in from the
 * whole roost to the ringed drone as the flight starts.
 */
const PAN = 3

/**
 * The seconds into the flight the pan ends on.
 */
const PANNED = PAN * PLAN.pace

/**
 * Eases `u` in and out, slow at either end and fastest halfway.
 */
const eased = (u: number): number => u < 0.5 ? 4 * u ** 3 : 1 - (2 - 2 * u) ** 3 / 2

/**
 * The stretches the loop runs through: the camera panning in from the whole
 * roost as the flight starts, the flight flown at its pace, held at its end,
 * faded out, and faded in again over the whole roost at its launch.
 */
const LOOP: readonly Stretch[] = [
  { lift: u => 1 - eased(u), span: PAN, t: u => u * PANNED },
  { span: FLOWN - PAN, t: u => PANNED + u * (PLAN.seconds - PANNED) },
  { span: 0.6, t: () => PLAN.seconds },
  { span: 0.8, t: () => PLAN.seconds, veil: u => 1 - eased(u) },
  { lift: () => 1, span: 0.8, t: () => 0, veil: eased }
]

/**
 * The whole loop's length, in seconds of the page's own time.
 */
const WHOLE = LOOP.reduce((sum, stretch) => sum + stretch.span, 0)

/**
 * The flight played on a loop that never cuts from its end to its launch: each
 * stretch carries the moment shown on, with how strongly the drawing shows and
 * how far the camera stands lifted at each. The still moment shows while
 * motion stands still.
 */
export class Player {
  readonly flight : Flight
  #at             = 0
  #lift           = 0
  #t              = 0
  #veil           = 1

  constructor(flight: Flight) {
    this.flight = flight
    flight.fly(1)
  }

  /**
   * Reads the moment shown, in seconds into the flight, flying the flight to
   * the still moment first while motion stands still.
   */
  get t(): number {
    if (!still()) return this.#t
    this.flight.fly(PLAN.still * PLAN.hz + 1)
    return PLAN.still
  }

  /**
   * Reads the last whole frame at or before the moment shown.
   */
  get frame(): number {
    return Math.min(Math.floor(this.t * PLAN.hz), FRAMES - 1)
  }

  /**
   * Reads how far the camera stands lifted toward the whole roost, from 0 to 1.
   */
  get lift(): number {
    return still() ? 0 : this.#lift
  }

  /**
   * Reads how strongly the drawing shows, from 0 to 1.
   */
  get veil(): number {
    return still() ? 1 : this.#veil
  }

  /**
   * Writes the place of the drone at `index` at the moment shown into `into`.
   */
  place(index: number, into: Float32Array): void {
    const { places } = this.flight
    const frame      = Math.min(this.frame, FRAMES - 2)
    const share      = this.t * PLAN.hz - frame
    const at         = (frame * this.flight.count + index) * 3
    const next       = at + this.flight.count * 3
    for (let k = 0; k < 3; k++) into[k] = common.lerp(places[at + k]!, places[next + k]!, share)
  }

  /**
   * Reads the value of `row`, which holds `stride` values a frame, at `offset`
   * of the moment shown.
   */
  read(row: Float32Array, stride = 1, offset = 0): number {
    const frame = Math.min(this.frame, FRAMES - 2)
    const share = this.t * PLAN.hz - frame
    return common.lerp(row[frame * stride + offset]!, row[(frame + 1) * stride + offset]!, share)
  }

  /**
   * Carries the loop on by `dt` seconds of the page's own time.
   */
  step(dt: number): void {
    this.#at   = (this.#at + dt) % WHOLE
    let from   = this.#at
    const shown = LOOP.find(stretch => (from -= stretch.span) < 0) ?? LOOP.at(-1)!
    const u    = Math.min(1, 1 + from / shown.span)
    this.#t    = shown.t(u)
    this.#veil = shown.veil?.(u) ?? 1
    this.#lift = shown.lift?.(u) ?? 0
    this.flight.fly(Math.floor(this.t * PLAN.hz) + 1)
  }
}
