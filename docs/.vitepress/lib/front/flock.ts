import type * as common    from '../common.ts'
import type { Composite }   from './composite.ts'
import { Flown }            from './flown.ts'
import { type Heat, repel } from './heat.ts'

/**
 * A flock as `Flock` builds it: `count` birds drawn from `seed`, each between
 * the two lengths `size` holds and drawing one of the `glyphs` traced birds,
 * the plan's launch spacing drawn `spacing` pixels long.
 */
interface Flight {
  count   : number
  /**
   * The least heat at a bird that pushes it, as a share of full strength.
   */
  faint   : number
  glyphs  : number
  /**
   * The most a heat at full strength pushes a bird across its flight, in
   * meters per second squared.
   */
  push    : number
  /**
   * The share of its region's half-width the roost's disk spans.
   */
  reach   : number
  seed    : number
  size    : readonly [number, number]
  spacing : number
  /**
   * The longest stretch one call to `Flock.step` flies, in seconds.
   */
  stretch : number
}

/**
 * How the site's flocks fly: from one seed, each bird between the two lengths
 * `size` holds, the plan's launch spacing drawn `spacing` pixels long, pushed
 * by heat from `faint` up to `push`, over a roost spanning `reach` of the
 * region, and no more than `stretch` seconds a frame.
 */
export const FLIGHT = {
  faint   : 0.002,
  push    : 60,
  reach   : 0.8,
  seed    : 41,
  size    : [4, 6.6],
  spacing : 7,
  stretch : 0.1
} as const

/**
 * A flock flown by the package's composite under its default slots, seen from
 * the side with east across and height up, the view fixed on the roost, whose
 * center and height sit at the middle of the flock's region and whose disk
 * spans most of the region's width, so the flock wanders the region under the
 * composite's own roost. Heat the field around a bird gives off pushes it
 * across its flight down the heat's slope, beside the slots, so the flock
 * breaks from the plume and from the falcon. Each bird's place and velocity in
 * the drawing's pixels sit in `x`, `y`, `vx`, and `vy`, and every array a step
 * reads and writes is built once, so a step allocates nothing.
 */
export class Flock {
  readonly glyph       : Uint8Array
  /**
   * Each bird's east, north, and height, in meters.
   */
  readonly positions   : Float64Array
  /**
   * The pixels one meter is drawn across.
   */
  readonly scale       : number
  readonly size        : Float32Array
  readonly vx          : Float32Array
  readonly vy          : Float32Array
  readonly x           : Float32Array
  readonly y           : Float32Array
  readonly #away       : [number, number] = [0, 0]
  #box                 : common.Box = { x0: 0, x1: 1, y0: 0, y1: 1 }
  #clock               = 0
  readonly #composite  : Composite
  readonly #flight     : Flight
  readonly #flown      : Flown
  readonly #pushes     : Float64Array

  constructor(flight: Flight) {
    const { count }  = flight
    this.#flown      = new Flown(count, flight.seed)
    this.#composite  = this.#flown.composite
    this.#flight     = flight
    this.#pushes     = new Float64Array(count * 3)
    this.positions   = this.#composite.positions
    this.scale       = flight.spacing / this.#composite.model.plan.spacing_m
    this.x           = new Float32Array(count)
    this.y           = new Float32Array(count)
    this.vx          = new Float32Array(count)
    this.vy          = new Float32Array(count)
    this.size        = new Float32Array(count)
    this.glyph       = new Uint8Array(count)
    const { random } = this.#flown
    for (let i = 0; i < count; i++) {
      this.size[i]  = flight.size[0] + (flight.size[1] - flight.size[0]) * random() ** 1.6
      this.glyph[i] = Math.floor(random() * flight.glyphs)
    }
    this.#flown.launch()
    this.#project()
  }

  /**
   * Flies the composite one step through the heat `field` gives off where each
   * bird is drawn.
   */
  #advance(field: Heat): void {
    const pushes = this.#pushes
    const away   = this.#away
    pushes.fill(0)
    for (let i = 0; i < this.x.length; i++) {
      const hot = field.heat(this.x[i]!, this.y[i]!)
      if (hot <= this.#flight.faint) continue
      repel(field, this.x[i]!, this.y[i]!, hot, this.#flight.push, away)
      pushes[i * 3]     = away[0]
      pushes[i * 3 + 2] = -away[1]
    }
    this.#flown.advance(pushes)
    this.#project()
  }

  /**
   * Draws each bird from the side, east across and height up, the roost's
   * center and height at the middle of the region.
   */
  #project(): void {
    const { positions, roost, velocities } = this.#composite
    const { x0, x1, y0, y1 }               = this.#box
    const across                           = (x0 + x1) / 2
    const down                             = (y0 + y1) / 2
    for (let at = 0, i = 0; i < this.x.length; at += 3, i++) {
      this.x[i]  = across + (positions[at]! - roost.center_m[0]) * this.scale
      this.y[i]  = down - (positions[at + 2]! - roost.height_m) * this.scale
      this.vx[i] = velocities[at]! * this.scale
      this.vy[i] = -velocities[at + 2]! * this.scale
    }
  }

  /**
   * Reads the distance between the birds at `first` and `second`, in the
   * drawing's pixels.
   */
  apart(first: number, second: number): number {
    const { positions } = this
    const a = first * 3, b = second * 3
    return this.scale * Math.hypot(
      positions[a]! - positions[b]!,
      positions[a + 1]! - positions[b + 1]!,
      positions[a + 2]! - positions[b + 2]!
    )
  }

  /**
   * Writes into `into` the senders the bird at `index` ranked at the last step,
   * nearest first by the positions its radio heard them at, `-1` past the last
   * it heard, and returns how many it ranked.
   */
  heard(index: number, into: Int32Array): number {
    const { model, ranks } = this.#composite
    const base             = index * model.plan.neighbors
    for (let rank = 0; rank < into.length; rank++) into[rank] = ranks[base + rank]!
    const gap = into.indexOf(-1)
    return gap < 0 ? into.length : gap
  }

  /**
   * Sets the region the flock is drawn in, the roost at its middle and the
   * roost's disk spanning the flight's `reach` of its half-width.
   */
  region(box: common.Box): void {
    this.#box                      = box
    this.#composite.roost.radius_m = this.#flight.reach * (box.x1 - box.x0) / 2 / this.scale
    this.#project()
  }

  /**
   * Runs the flock ahead by `seconds` without drawing, so the first frame shows
   * a formed flock rather than its launch.
   */
  settle(seconds: number, field: Heat): void {
    for (let t = 0; t < seconds; t += this.#composite.model.plan.step_s) this.#advance(field)
  }

  /**
   * Moves the flock on by `dt` seconds through the heat `field` gives off, in
   * whole steps of the plan's `step_s`, carrying the remainder to the next
   * call.
   */
  step(dt: number, field: Heat): void {
    const step   = this.#composite.model.plan.step_s
    this.#clock += Math.min(dt, this.#flight.stretch)
    for (; this.#clock >= step; this.#clock -= step) this.#advance(field)
  }
}
