import * as common           from '../common.ts'
import { Composite, ranked } from './composite.ts'
import { type Heat, repel }  from './heat.ts'

/**
 * A flock as `Flock` builds it: `count` birds drawn from `seed`, each between
 * the two lengths `size` holds and drawing one of the `glyphs` traced birds, the
 * plan's launch spacing drawn `spacing` pixels long.
 */
interface Flight {
  count   : number
  glyphs  : number
  seed    : number
  size    : readonly [number, number]
  spacing : number
}

/**
 * The most a heat at full strength pushes a bird across its flight, in meters
 * per second squared.
 */
const BREAK = 60
/**
 * The longest stretch one call to `Flock.step` flies, in seconds, so a frame
 * after a stall runs a bounded number of steps.
 */
const STRETCH = 0.1

/**
 * How the site's flocks fly: from one seed, each bird between the two lengths
 * `size` holds, the plan's launch spacing drawn `spacing` pixels long.
 */
export const FLIGHT = { seed: 41, size: [4, 6.6], spacing: 7 } as const

/**
 * A flock flown by the package's composite under its default slots, seen from
 * the side with east across and height up, the view following the middle of
 * the flock so it stays in the middle of its region. Heat the field around a
 * bird gives off pushes it across its flight down the heat's slope, beside the
 * slots, so the flock breaks from the plume and from the falcon. Each bird's
 * place and velocity in the drawing's pixels sit in `x`, `y`, `vx`, and `vy`,
 * and every array a step reads and writes is built once, so a step allocates
 * nothing.
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
  readonly #increments : Float64Array
  readonly #normal     : () => number
  readonly #pushes     : Float64Array

  constructor(flight: Flight) {
    const { count } = flight
    const random    = common.mulberry(flight.seed)
    this.#composite  = new Composite(count)
    this.#normal     = common.gaussian(random)
    this.#increments = new Float64Array(count * 3)
    this.#pushes     = new Float64Array(count * 3)
    this.positions   = this.#composite.positions
    this.scale       = flight.spacing / this.#composite.model.plan.spacing_m
    this.x           = new Float32Array(count)
    this.y           = new Float32Array(count)
    this.vx          = new Float32Array(count)
    this.vy          = new Float32Array(count)
    this.size        = new Float32Array(count)
    this.glyph       = new Uint8Array(count)
    for (let i = 0; i < count; i++) {
      this.size[i]  = flight.size[0] + (flight.size[1] - flight.size[0]) * random() ** 1.6
      this.glyph[i] = Math.floor(random() * flight.glyphs)
    }
    this.#composite.launch(this.#normal)
    this.#project()
  }

  /**
   * Flies the composite one step through the heat `field` gives off where each
   * bird is drawn, under fresh Brownian increments.
   */
  #advance(field: Heat): void {
    const root   = Math.sqrt(this.#composite.model.plan.step_s)
    const pushes = this.#pushes
    const away   = this.#away
    for (let i = 0; i < this.#increments.length; i++) this.#increments[i] = this.#normal() * root
    pushes.fill(0)
    for (let i = 0; i < this.x.length; i++) {
      const hot = field.heat(this.x[i]!, this.y[i]!)
      if (hot <= 0.002) continue
      repel(field, this.x[i]!, this.y[i]!, hot, BREAK, away)
      pushes[i * 3]     = away[0]
      pushes[i * 3 + 2] = -away[1]
    }
    this.#composite.advance(this.#increments, pushes)
    this.#project()
  }

  /**
   * Draws each bird from the side, east across and height up, the flock's
   * middle at the middle of the region.
   */
  #project(): void {
    const { positions, velocities } = this.#composite
    const { x0, x1, y0, y1 }        = this.#box
    const count                     = this.x.length
    let east                        = 0, up = 0
    for (let at = 0; at < positions.length; at += 3) {
      east += positions[at]!
      up   += positions[at + 2]!
    }
    for (let at = 0, i = 0; i < count; at += 3, i++) {
      this.x[i]  = (x0 + x1) / 2 + (positions[at]! - east / count) * this.scale
      this.y[i]  = (y0 + y1) / 2 - (positions[at + 2]! - up / count) * this.scale
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
    return this.scale * Math.hypot(
      positions[first * 3]! - positions[second * 3]!,
      positions[first * 3 + 1]! - positions[second * 3 + 1]!,
      positions[first * 3 + 2]! - positions[second * 3 + 2]!
    )
  }

  /**
   * Writes the indices of the bird at `index`'s `best.length` nearest others
   * into `into`, nearest first, by the distance between the drones the flock
   * flies, leaving in `best` the squared distances in meters.
   */
  nearest(best: Float64Array, index: number, into: Int32Array): void {
    ranked(best, index, into, this.positions)
  }

  /**
   * Sets the region the flock is drawn in, its middle at the region's middle.
   */
  region(box: common.Box): void {
    this.#box = box
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
   * whole steps of the plan's `step_s`, carrying the remainder to the next call.
   */
  step(dt: number, field: Heat): void {
    const step   = this.#composite.model.plan.step_s
    this.#clock += Math.min(dt, STRETCH)
    for (; this.#clock >= step; this.#clock -= step) this.#advance(field)
  }
}
