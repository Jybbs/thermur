import * as common          from '../common.ts'
import { type Heat, repel } from './heat.ts'

/**
 * A flock as `Flock` builds it: `count` birds drawn from `seed`, each between
 * the two lengths `size` holds and drawing one of the `glyphs` traced birds,
 * cruising at `speed` and keeping `sep` pixels from its nearest neighbors, and
 * following the roost `roost` writes into its last argument for each moment and
 * region.
 */
interface Flight {
  count  : number
  glyphs : number
  roost  : (t: number, box: common.Box, out: [number, number]) => void
  seed   : number
  sep    : number
  size   : readonly [number, number]
  speed  : number
}

/**
 * A murmuration of birds that each read their seven nearest neighbors, align
 * with them, keep together and apart, follow a roost that wanders the sky, and
 * turn away from whatever heat the field around them gives off, an alarm
 * passing from neighbor to neighbor once one of them feels it. Every array a
 * step reads and writes is built once, so a step allocates nothing.
 */
export class Flock {
  readonly glyph   : Uint8Array
  readonly size    : Float32Array
  readonly vx      : Float32Array
  readonly vy      : Float32Array
  readonly x       : Float32Array
  readonly y       : Float32Array
  readonly #away   : [number, number] = [0, 0]
  readonly #ax     : Float32Array
  readonly #ay     : Float32Array
  readonly #best   = new Float32Array(common.NEIGHBORS)
  #box             : common.Box = { x0: 0, x1: 1, y0: 0, y1: 1 }
  #calm            = 0
  readonly #flight : Flight
  #margin          = 0
  readonly #nbr    : Int16Array
  readonly #panic  : Float32Array
  #placed          = false
  readonly #random : () => number
  readonly #roost  : [number, number] = [0, 0]
  #t               = 0

  constructor(flight: Flight) {
    const { count } = flight
    this.#flight    = flight
    this.#random    = common.mulberry(flight.seed)
    this.x          = new Float32Array(count)
    this.y          = new Float32Array(count)
    this.vx         = new Float32Array(count)
    this.vy         = new Float32Array(count)
    this.#ax        = new Float32Array(count)
    this.#ay        = new Float32Array(count)
    this.size       = new Float32Array(count)
    this.glyph      = new Uint8Array(count)
    this.#panic     = new Float32Array(count)
    this.#nbr       = new Int16Array(count * common.NEIGHBORS)
    for (let i = 0; i < count; i++) {
      this.size[i]  = flight.size[0] + (flight.size[1] - flight.size[0]) * this.#random() ** 1.6
      this.glyph[i] = Math.floor(this.#random() * flight.glyphs)
    }
  }

  /**
   * Moves every bird on by `dt` seconds along the force `#steer` set on it,
   * easing its speed toward the flock's cruise, faster while it panics.
   */
  #move(dt: number): void {
    const { vx, vy, x, y } = this
    const ax               = this.#ax
    const ay               = this.#ay
    const panic            = this.#panic
    for (let i = 0; i < this.#flight.count; i++) {
      vx[i]   = vx[i]! + ax[i]! * dt
      vy[i]   = vy[i]! + ay[i]! * dt
      const s = Math.hypot(vx[i]!, vy[i]!) + 1e-6
      const k = 1 + (this.#flight.speed * (1 + panic[i]! * 0.9) / s - 1) * Math.min(1, dt * 2.5)
      vx[i]   = vx[i]! * k
      vy[i]   = vy[i]! * k
      x[i]    = x[i]! + vx[i]! * dt
      y[i]    = y[i]! + vy[i]! * dt
    }
  }

  /**
   * Finds each bird's seven nearest neighbors by distance alone, however far
   * they stand.
   */
  #neighbors(): void {
    for (let i = 0; i < this.#flight.count; i++) {
      common.nearest(this.#best, i, this.#nbr, this.x, this.y, i * common.NEIGHBORS)
    }
  }

  /**
   * Scatters the birds around the roost's first place, all headed one way.
   */
  #scatter(): void {
    const { x0, x1, y0, y1 } = this.#box
    this.#flight.roost(0, this.#box, this.#roost)
    const [cx, cy] = this.#roost
    const reach    = Math.min(x1 - x0, y1 - y0) * 0.18
    const heading  = this.#random() * Math.PI * 2
    for (let i = 0; i < this.#flight.count; i++) {
      const angle  = this.#random() * Math.PI * 2
      const radius = Math.sqrt(this.#random()) * reach
      this.x[i]    = cx + Math.cos(angle) * radius * 1.6
      this.y[i]    = cy + Math.sin(angle) * radius * 0.7
      this.vx[i]   = Math.cos(heading) * this.#flight.speed
      this.vy[i]   = Math.sin(heading) * this.#flight.speed
    }
    this.#placed = true
  }

  /**
   * Sets the force on bird `i`: toward its neighbors' heading and middle and
   * apart from the nearest, toward the roost, back from the region's edges, and
   * down the slope of the heat it feels, its panic rising with the heat and the
   * alarm its neighbors pass on.
   */
  #steer(i: number, dt: number, field: Heat): void {
    const { vx, vy, x, y }   = this
    const { x0, x1, y0, y1 } = this.#box
    const panic              = this.#panic
    const away               = this.#away
    const reach              = this.#flight.sep * (1 + panic[i]! * 1.4)
    let alarm                = 0, apx = 0, apy = 0, avx = 0, avy = 0, sx = 0, sy = 0
    for (let k = 0; k < common.NEIGHBORS; k++) {
      const j  = this.#nbr[i * common.NEIGHBORS + k]!
      const dx = x[i]! - x[j]!
      const dy = y[i]! - y[j]!
      const d  = Math.sqrt(dx * dx + dy * dy + 0.01)
      avx     += vx[j]!
      avy     += vy[j]!
      apx     += x[j]!
      apy     += y[j]!
      alarm    = Math.max(alarm, panic[j]!)
      sx      += d < reach ? dx / d * (reach - d) / reach : 0
      sy      += d < reach ? dy / d * (reach - d) / reach : 0
    }
    const tx   = this.#roost[0] - x[i]!
    const ty   = this.#roost[1] - y[i]!
    const td   = Math.hypot(tx, ty) + 1
    const pull = common.clamp((td - this.#calm) * 1.4, 0, 240)
    const hot  = field.heat(x[i]!, y[i]!)
    let fx     = (avx / common.NEIGHBORS - vx[i]!) * 2.4 + (apx / common.NEIGHBORS - x[i]!) * 1.6 + sx * 240
               + (this.#random() - 0.5) * 60 + tx / td * pull
               + Math.max(0, x0 + this.#margin - x[i]!) * 6 - Math.max(0, x[i]! - x1 + this.#margin) * 6
    let fy = (avy / common.NEIGHBORS - vy[i]!) * 2.4 + (apy / common.NEIGHBORS - y[i]!) * 1.6 + sy * 240
           + (this.#random() - 0.5) * 60 + ty / td * pull
           + Math.max(0, y0 + this.#margin - y[i]!) * 6 - Math.max(0, y[i]! - y1 + this.#margin) * 6
    if (hot > 0.002) {
      repel(field, x[i]!, y[i]!, hot, 900, away)
      fx += away[0]
      fy += away[1] * 0.4
    }
    panic[i]    = Math.max(Math.min(1, hot * 3), alarm * 0.86, panic[i]! - dt * 0.9)
    this.#ax[i] = fx
    this.#ay[i] = fy
  }

  /**
   * Sets the region the birds keep to, scattering them on the first call and
   * moving every bird with the region on each later one.
   */
  region(box: common.Box): void {
    const old = this.#box
    this.#box = box
    if (!this.#placed) {
      this.#scatter()
      return
    }
    const sx = (box.x1 - box.x0) / Math.max(1, old.x1 - old.x0)
    const sy = (box.y1 - box.y0) / Math.max(1, old.y1 - old.y0)
    for (let i = 0; i < this.#flight.count; i++) {
      this.x[i] = box.x0 + (this.x[i]! - old.x0) * sx
      this.y[i] = box.y0 + (this.y[i]! - old.y0) * sy
    }
  }

  /**
   * Runs the flock ahead by `seconds` without drawing, so the first frame shows
   * a formed flock rather than its first scatter.
   */
  settle(seconds: number, field: Heat): void {
    for (let t = 0; t < seconds; t += 1 / 30) this.step(1 / 30, field)
  }

  /**
   * Moves every bird on by `dt` seconds through the heat `field` gives off.
   */
  step(dt: number, field: Heat): void {
    const { x0, x1, y0, y1 } = this.#box
    this.#t                 += dt
    this.#margin             = Math.min(x1 - x0, y1 - y0) * 0.08
    this.#calm               = Math.min(x1 - x0, y1 - y0) * 0.14
    this.#neighbors()
    this.#flight.roost(this.#t, this.#box, this.#roost)
    for (let i = 0; i < this.#flight.count; i++) this.#steer(i, dt, field)
    this.#move(dt)
  }
}
