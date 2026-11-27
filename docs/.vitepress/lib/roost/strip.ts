import type { Geometry, Part } from '../canvas/part.ts'
import type { Tones }          from '../canvas/tones.ts'
import { seriesOf }            from '../identity.ts'
import { FRAMES, PLAN }        from './flight.ts'
import type { Player }         from './player.ts'

/**
 * One reading of the flight: its name, the row it reads, the range its strip
 * spans, how its value is written, what it is held against in words, and that
 * reference's range on the strip, one value drawn as a line.
 */
export interface Reading {
  against : string
  legend? : string
  band    : readonly [number, number]
  name    : string
  of      : (player: Player) => Float32Array
  span    : readonly [number, number]
  write   : (value: number) => string
  edges?  : (player: Player) => readonly [Float32Array, Float32Array]
}

/**
 * The three readings the view holds beside the flight, the polarization
 * against the starling flocks' and the spacing against the outdoor drone
 * flock's, each as the site's methods cite it, and the height against the
 * roost's own.
 */
export const READINGS: readonly Reading[] = [
  {
    against : 'Starling flocks 0.96',
    band    : [0.96, 0.96],
    name    : 'Polarization',
    of      : player => player.flight.polarization,
    span    : [0, 1.05],
    write   : value => value.toFixed(2)
  },
  {
    against : 'Outdoor drones 12 to 30 m',
    band    : [12, 30],
    name    : 'Spacing',
    of      : player => player.flight.spacing,
    span    : [0, 36],
    write   : value => `${value.toFixed(1)} m`
  },
  {
    against : 'Roost 100 m',
    band    : [100, 100],
    legend  : 'Roost 100 m, lowest to highest in gray',
    edges   : player => [player.flight.low, player.flight.high],
    name    : 'Height',
    of      : player => player.flight.height,
    span    : [0, 200],
    write   : value => `${value.toFixed(0)} m`
  }
]

/**
 * One reading drawn over the whole flight's time, its line running up to the
 * moment shown over the reference it is held against, the height's line inside
 * the lowest and highest drone's.
 */
export class Strip implements Part {
  readonly #player  : Player
  readonly #reading : Reading
  readonly #tones   : Tones

  constructor(player: Player, reading: Reading, tones: Tones) {
    this.#player  = player
    this.#reading = reading
    this.#tones   = tones
  }

  /**
   * Draws the strip's baseline with a tick every minute, and what the reading
   * is held against, a dashed line for one value or a band for a range.
   */
  #reference(ctx: CanvasRenderingContext2D, g: Geometry, x: (frame: number) => number, y: (value: number) => number): void {
    const { band } = this.#reading
    const held     = seriesOf(1, g.scheme)
    ctx.strokeStyle = this.#tones.ink(g.scheme, 0.25)
    ctx.lineWidth   = 1
    ctx.beginPath()
    ctx.moveTo(0, g.h - 0.5)
    ctx.lineTo(g.w, g.h - 0.5)
    for (let t = 0; t <= PLAN.seconds; t += 60) {
      const at = Math.round(x(t * PLAN.hz)) + 0.5
      ctx.moveTo(at, g.h)
      ctx.lineTo(at, g.h - 5)
    }
    ctx.stroke()
    ctx.fillStyle   = held
    ctx.strokeStyle = held
    if (band[0] === band[1]) {
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(0, y(band[0]))
      ctx.lineTo(g.w, y(band[0]))
      ctx.stroke()
      ctx.setLineDash([])
    } else {
      ctx.globalAlpha = 0.18
      ctx.fillRect(0, y(band[1]), g.w, y(band[0]) - y(band[1]))
      ctx.globalAlpha = 1
    }
  }

  paint(ctx: CanvasRenderingContext2D, g: Geometry): void {
    const { edges, of, span } = this.#reading
    const row  = of(this.#player)
    const last = this.#player.frame
    const x    = (frame: number): number => frame / (FRAMES - 1) * (g.w - 8) + 4
    const y    = (value: number): number => g.h - 1 - (value - span[0]) / (span[1] - span[0]) * (g.h - 2)
    const step = Math.max(1, Math.floor(FRAMES / Math.max(1, g.w * 1.5)))
    this.#reference(ctx, g, x, y)
    if (edges) {
      const [low, high] = edges(this.#player)
      ctx.fillStyle     = this.#tones.ink(g.scheme, 0.14)
      ctx.beginPath()
      for (let frame = 0; frame <= last; frame += step) ctx.lineTo(x(frame), y(high[frame]!))
      for (let frame = last - last % step; frame >= 0; frame -= step) ctx.lineTo(x(frame), y(low[frame]!))
      ctx.fill()
    }
    ctx.strokeStyle = this.#tones.ink(g.scheme, 1)
    ctx.lineWidth   = 1.6
    ctx.lineJoin    = 'round'
    ctx.beginPath()
    for (let frame = 0; frame <= last; frame += step) ctx.lineTo(x(frame), y(row[frame]!))
    ctx.lineTo(x(last), y(row[last]!))
    ctx.stroke()
    ctx.fillStyle = this.#tones.flock(g.scheme, 1, 1)
    ctx.beginPath()
    ctx.arc(x(last), y(row[last]!), 3.5, 0, Math.PI * 2)
    ctx.fill()
  }
}
