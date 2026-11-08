import type { Flame, Marks }     from '../canvas/marks.ts'
import type { Geometry, Part }   from '../canvas/part.ts'
import type { Tones }            from '../canvas/tones.ts'
import * as common               from '../common.ts'
import * as flock                from '../front/flock.ts'
import { joined }                from '../front/heat.ts'
import { Plume, RISE }           from '../front/plume.ts'
import { PointerHeat }           from '../front/pointer-heat.ts'
import { type Identity, type Scheme, fontStack, seriesOf } from '../identity.ts'
import * as paint                from './paint.ts'

/**
 * What the pointer does over the sky: chase the flock as the falcon, or mark
 * the bird it presses.
 */
export type Mode = (typeof MODES)[number]
/**
 * What the controls set and the drawing reads: the marked bird, how far its
 * lines to its seven have grown, what the pointer does, and the fixed range,
 * in bird lengths.
 */
export interface Shared {
  chosen : number
  grown  : number
  mode   : Mode
  range  : number
}
/**
 * How the drawing lays itself out and moves: the fewest and most birds it
 * flies, the fire's strength they fly from, how tall the strips run along its
 * foot, how far above them the flock's region stops, the sky each bird flies
 * in, in the flock's own square pixels, the samples the trace takes a second,
 * the pixels within which the pointer reaches a bird, the milliseconds between
 * two writes of the readout, the seconds the flock flies unseen before the
 * first frame, the seconds the trace keeps, and how large the sky draws the
 * flock.
 */
export interface Tuning {
  birds  : readonly [number, number]
  fire   : number
  foot   : number
  inset  : number
  per    : number
  rate   : number
  reach  : number
  read   : number
  settle : number
  trace  : number
  zoom   : number
}

/**
 * Where along the flock's gradient, from the ink at its foot to the fire at its
 * head, the marked bird and its seven are drawn.
 */
export const ALONG = { marked: 1, seven: 0.62 } as const
/**
 * What the pointer does over the sky, the first its start.
 */
export const MODES = ['predator', 'select'] as const
/**
 * The place in the palette's chart series of the color the drawing gives the
 * fixed range: its ring, the birds inside it, and its count in the trace.
 */
const RANGED = 1
/**
 * The opacity of a bird neither among the marked bird's seven nor inside the
 * range, drawn in the ink at the gradient's foot so it reads over either sky.
 */
const REST = 0.5
/**
 * The front page's own flock, flown larger in the same sky under the same
 * roost and plume, its birds each reading their seven nearest. One bird is
 * marked with a hairline to its seven and a dashed ring at a fixed range, and
 * two strips along the foot count the birds inside the ring over the last
 * seconds against the seven by rank, which holds as the flock gathers and
 * splits. The pointer is the falcon in predator mode and marks the bird it
 * presses in select mode, where the flock flies on undisturbed.
 */
export class Falcon implements Part {
  readonly #best    = new Float32Array(common.NEIGHBORS)
  #clock            = 0
  readonly #falcon  = new PointerHeat()
  readonly #fire    = new Plume(RISE)
  readonly #flame   : Flame = { fire: 0, half: 18, scheme: 'dark', x: 0, y: 0 }
  readonly #field   = joined(this.#fire, this.#falcon)
  #flock            : flock.Flock | null = null
  #floor            = 0
  readonly #marks   : Marks
  readonly #nbr     = new Int16Array(common.NEIGHBORS)
  readonly #read    : (seventh: number, scheme: Scheme) => void
  readonly #shared  : Shared
  #stamp            = -Infinity
  readonly #strips  : paint.Strips
  readonly #tones   : Tones
  readonly #tuning  : Tuning
  #within           = 0

  constructor(
    identity : Identity,
    marks    : Marks,
    read     : (seventh: number, scheme: Scheme) => void,
    shared   : Shared,
    tones    : Tones,
    tuning   : Tuning
  ) {
    const font       = `500 11px ${fontStack(identity.faces.mono)}`
    this.#marks      = marks
    this.#read       = read
    this.#shared     = shared
    this.#tones      = tones
    this.#tuning     = tuning
    this.#strips     = new paint.Strips(font, tuning.trace * tuning.rate, tuning.trace, tones)
    this.#fire.fire  = tuning.fire
    this.#flame.fire = tuning.fire
  }

  /**
   * Reads the mean length of a bird, in the flock's own pixels.
   */
  get #length(): number {
    return (flock.FLIGHT.size[0] + flock.FLIGHT.size[1]) / 2
  }

  /**
   * Reads the distance to the marked bird's seventh, in bird lengths.
   */
  get #seventh(): number {
    return Math.sqrt(this.#best[common.NEIGHBORS - 1]!) / this.#length
  }

  /**
   * Counts the birds inside the fixed range of the marked one and finds its
   * seven nearest.
   */
  #count(): void {
    const { x, y }          = this.#flock!
    const { chosen, range } = this.#shared
    const reach             = range * this.#length
    common.nearest(this.#best, chosen, this.#nbr, x, y)
    let within = 0
    for (let index = 0; index < x.length; index++) {
      if (index !== chosen && Math.hypot(x[index]! - x[chosen]!, y[index]! - y[chosen]!) <= reach) within++
    }
    this.#within = within
  }

  /**
   * Finds the bird nearest the flock's middle.
   */
  #middle(): number {
    const { x, y } = this.#flock!
    const cx       = x.reduce((sum, each) => sum + each, 0) / x.length
    const cy       = y.reduce((sum, each) => sum + each, 0) / y.length
    let best       = 0
    for (let index = 1; index < x.length; index++) {
      if (Math.hypot(x[index]! - cx, y[index]! - cy) < Math.hypot(x[best]! - cx, y[best]! - cy)) best = index
    }
    return best
  }

  /**
   * Draws the flock: the marked bird's hairlines out to its seven as far as
   * they have grown, every other bird in the range's color inside the range,
   * in the seven's among its seven, and faint in the ink elsewhere, and the
   * marked bird over them.
   */
  #paintFlock(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number, ranged: string): void {
    const birds                    = this.#flock!
    const { x, y }                 = birds
    const { chosen, grown, range } = this.#shared
    const zoom                     = this.#tuning.zoom
    const reach                    = range * this.#length
    const ox                       = x[chosen]! * zoom
    const oy                       = y[chosen]! * zoom
    ctx.strokeStyle                = this.#tones.flock(g.scheme, ALONG.seven, 0.7)
    ctx.lineWidth                  = 1.2
    ctx.beginPath()
    for (let rank = 0; rank < common.NEIGHBORS; rank++) {
      const index = this.#nbr[rank]!
      ctx.moveTo(ox, oy)
      ctx.lineTo(ox + (x[index]! * zoom - ox) * grown, oy + (y[index]! * zoom - oy) * grown)
    }
    ctx.stroke()
    for (let index = 0; index < x.length; index++) {
      if (index === chosen) continue
      const seven  = this.#nbr.includes(index)
      const inside = Math.hypot(x[index]! - x[chosen]!, y[index]! - y[chosen]!) <= reach
      const color  = inside ? ranged : this.#tones.flock(g.scheme, seven ? ALONG.seven : 0, seven ? 1 : REST)
      paint.bird(ctx, birds, color, dpr, index, this.#marks, seven ? 1.15 : 1, zoom)
    }
    paint.bird(ctx, birds, this.#tones.flock(g.scheme, ALONG.marked, 1), dpr, chosen, this.#marks, 1.6, zoom)
  }

  /**
   * Finds the bird nearest the pointer within reach, or -1.
   */
  #under(g: Geometry): number {
    if (g.pointer === null) return -1
    const { x, y }        = this.#flock!
    const { reach, zoom } = this.#tuning
    let best              = -1
    let gap               = reach * reach
    for (let index = 0; index < x.length; index++) {
      const distance = (x[index]! * zoom - g.pointer[0]) ** 2 + (y[index]! * zoom - g.pointer[1]) ** 2
      if (distance < gap) {
        gap  = distance
        best = index
      }
    }
    return best
  }

  /**
   * Splits the canvas into the sky and the strips along its foot, sets the
   * flock's region to the sky short of the strips by the inset and the fire on
   * the sky's foot, and builds and settles a flock sized to the sky where its
   * size calls for a different count, marking the bird nearest its middle.
   */
  fit(g: Geometry): void {
    const { birds, foot, inset, per, zoom } = this.#tuning
    const floor                             = g.h - foot
    const box                               = { x0: 0, x1: g.w / zoom, y0: 0, y1: (floor - inset) / zoom }
    const fits                              = common.clamp(box.x1 * box.y1 / per, ...birds)
    const count                             = Math.round(fits / 10) * 10
    this.#floor                             = floor
    this.#strips.panel                      = { x0: 0, x1: g.w, y0: floor, y1: g.h }
    this.#fire.x                            = box.x1 * 0.76
    this.#fire.y                            = floor / zoom
    this.#flame.x                           = this.#fire.x * zoom
    this.#flame.y                           = floor - 3
    this.#flock?.region(box)
    if (this.#flock?.x.length === count) return
    this.#flock         = new flock.Flock({
      ...flock.FLIGHT,
      count,
      glyphs : this.#marks.count,
      roost  : flock.roost
    })
    this.#flock.region(box)
    this.#flock.settle(this.#tuning.settle, this.#field)
    this.#shared.chosen = this.#middle()
    this.#count()
    this.#strips.fill(this.#within)
  }

  paint(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number): void {
    const { x, y }                = this.#flock!
    const { chosen, mode, range } = this.#shared
    const zoom                    = this.#tuning.zoom
    const ranged                  = seriesOf(RANGED, g.scheme)
    const hovered                 = mode === 'select' ? this.#under(g) : -1
    this.#flame.scheme            = g.scheme
    this.#count()
    paint.sky(ctx, g, this.#tones)
    this.#marks.fire(ctx, this.#flame)
    paint.ring(ctx, 0.16, ranged, range * this.#length * zoom, x[chosen]! * zoom, y[chosen]! * zoom)
    this.#paintFlock(ctx, g, dpr, ranged)
    if (hovered >= 0 && hovered !== chosen) {
      ctx.strokeStyle = this.#tones.ink(g.scheme, 0.8)
      ctx.lineWidth   = 1.2
      ctx.beginPath()
      ctx.arc(x[hovered]! * zoom, y[hovered]! * zoom, 9, 0, Math.PI * 2)
      ctx.stroke()
    }
    this.#strips.paint(ctx, g, ranged, this.#tones.flock(g.scheme, ALONG.seven, 1))
    const now = performance.now()
    if (now - this.#stamp < this.#tuning.read && !g.still) return
    this.#stamp = now
    this.#read(this.#seventh, g.scheme)
  }

  press(g: Geometry): void {
    if (this.#shared.mode !== 'select') return
    const index = this.#under(g)
    if (index >= 0) this.#shared.chosen = index
  }

  /**
   * Sways the plume, puts the falcon at the pointer while predator mode is on
   * and the pointer is over the sky, moves the flock on, and samples the count
   * inside the range.
   */
  step(dt: number, g: Geometry): void {
    const { pointer }    = g
    const { rate, zoom } = this.#tuning
    this.#fire.t        += dt
    this.#falcon.on      = this.#shared.mode === 'predator' && pointer !== null && pointer[1] < this.#floor
    this.#falcon.x       = (pointer?.[0] ?? 0) / zoom
    this.#falcon.y       = (pointer?.[1] ?? 0) / zoom
    this.#flock!.step(dt, this.#field)
    this.#clock         += dt
    while (this.#clock >= 1 / rate) {
      this.#clock -= 1 / rate
      this.#count()
      this.#strips.sample(this.#within)
    }
  }
}
