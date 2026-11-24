
import type { Geometry, Part } from '../canvas/part.ts'
import type { Tones }          from '../canvas/tones.ts'
import source                  from '../rendered/composite.json' with { type: 'json' }
import type { Playing }        from './flight.ts'
import { Measures }            from './measures.ts'

/**
 * The boxes the Lookout writes its readings into: the drone count, the
 * simulated time, the polarization, and the number of groups.
 */
export interface Readout {
  drones       : HTMLElement
  groups       : HTMLElement
  polarization : HTMLElement
  time         : HTMLElement
}

/**
 * How the Lookout draws a flight: each drone's side in pixels, the share of
 * the drawing's half-width and half-height the flock's spread from its middle
 * fills at most, how many of the spread's deviations that share spans, the
 * rate per second the view glides toward the flock at, and the seconds between
 * two readings.
 */
export const VIEW = { dot: 2, fill: 0.8, glide: 2, read: 0.25, spread: 3 } as const

/**
 * The flight a Lookout plays from the side, east across and height up, its view
 * gliding to keep the flock's middle at the middle of the drawing and its
 * spread inside it, every drone drawn as one dot in the scheme's ink. Beside
 * the flight it reads the drone count, the time flown, and the measures the
 * composite reports for it into `readout` as the flight plays.
 */
export class Lookout implements Part {
  #across           = 0
  #measures         : Measures
  #playing          : Playing
  readonly #readout : Readout
  #scale            = 0
  #since            = Infinity
  readonly #tones   : Tones
  #up               = 0

  constructor(playing: Playing, readout: Readout, tones: Tones) {
    this.#playing  = playing
    this.#readout  = readout
    this.#tones    = tones
    this.#measures = new Measures(playing.count, 3 * source.plan.spacing_m)
  }

  /**
   * Finds the view the flock's middle and spread set in a drawing `w` by `h`,
   * as the east and the height at its middle and the pixels one meter spans.
   */
  #aim(w: number, h: number): [number, number, number] {
    const { count, positions } = this.#playing
    let east = 0, east2 = 0, up = 0, up2 = 0
    for (let at = 0; at < positions.length; at += 3) {
      east  += positions[at]!
      up    += positions[at + 2]!
      east2 += positions[at]! ** 2
      up2   += positions[at + 2]! ** 2
    }
    const [x, z] = [east / count, up / count]
    const wide   = Math.sqrt(Math.max(east2 / count - x * x, 0)) * VIEW.spread
    const tall   = Math.sqrt(Math.max(up2 / count - z * z, 0)) * VIEW.spread
    return [x, z, VIEW.fill * Math.min(w / 2 / Math.max(wide, 1), h / 2 / Math.max(tall, 1))]
  }

  /**
   * Writes the drone count, the time flown, and the measures into the readout.
   */
  #read(): void {
    const { positions, seconds, velocities } = this.#playing
    this.#readout.drones.textContent       = this.#playing.count.toLocaleString('en-US')
    this.#readout.time.textContent         = `${seconds.toFixed(1)} s`
    this.#readout.polarization.textContent = this.#measures.polarization(velocities).toFixed(2)
    this.#readout.groups.textContent       = String(this.#measures.groups(positions))
  }

  fit(g: Pick<Geometry, 'h' | 'w'>): void {
    [this.#across, this.#up, this.#scale] = this.#aim(g.w, g.h)
    this.#read()
  }

  paint(ctx: CanvasRenderingContext2D, g: Geometry): void {
    const { positions } = this.#playing
    const half          = VIEW.dot / 2
    ctx.fillStyle       = this.#tones.ink(g.scheme, 0.85)
    for (let at = 0; at < positions.length; at += 3) {
      const x = g.w / 2 + (positions[at]! - this.#across) * this.#scale
      const y = g.h / 2 - (positions[at + 2]! - this.#up) * this.#scale
      ctx.fillRect(x - half, y - half, VIEW.dot, VIEW.dot)
    }
  }

  /**
   * Plays `playing` in place of the flight before, from its own start.
   */
  play(playing: Playing, g: Pick<Geometry, 'h' | 'w'>): void {
    this.#playing  = playing
    this.#measures = new Measures(playing.count, 3 * source.plan.spacing_m)
    this.#since    = Infinity
    this.fit(g)
  }

  step(dt: number, g: Geometry): void {
    this.#playing.play(dt)
    const [across, up, scale] = this.#aim(g.w, g.h)
    const share               = 1 - Math.exp(-VIEW.glide * dt)
    this.#across += share * (across - this.#across)
    this.#up     += share * (up - this.#up)
    this.#scale  += share * (scale - this.#scale)
    this.#since  += dt
    if (this.#since < VIEW.read) return
    this.#since = 0
    this.#read()
  }
}
