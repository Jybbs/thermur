import type { Marks }    from '../canvas/marks.ts'
import type { Geometry } from '../canvas/part.ts'
import type { Tones }    from '../canvas/tones.ts'
import type { Flock }    from '../front/flock.ts'
import * as common       from '../common.ts'

/**
 * A rectangle on the canvas, in pixels.
 */
export interface Rect {
  x0 : number
  x1 : number
  y0 : number
  y1 : number
}

/**
 * The dash of the ring at the fixed range, in pixels.
 */
const DASH = [5, 4]
/**
 * The unbroken line the ring returns to from `DASH`.
 */
const LINE: readonly number[] = []

/**
 * Draws the bird at `index` of `birds` centered on where the flock places it
 * rather than by its head, `scale` times its size and `zoom` times the
 * flock's, turned along its flight, in `color`, then restores the frame's
 * transform.
 */
export function bird(
  ctx   : CanvasRenderingContext2D,
  birds : Flock,
  color : string,
  dpr   : number,
  index : number,
  marks : Marks,
  scale : number,
  zoom  : number
): void {
  const angle  = Math.atan2(birds.vy[index]!, birds.vx[index]!)
  const length = birds.size[index]! * zoom * scale
  const x      = birds.x[index]! * zoom + Math.cos(angle) * length / 2
  const y      = birds.y[index]! * zoom + Math.sin(angle) * length / 2
  marks.bird(ctx, dpr, birds.glyph[index]!, x, y, angle, length, color)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
}

/**
 * Strokes a dashed ring `radius` pixels around a point in `color`, its inside
 * washed with the same color at `alpha`.
 */
export function ring(
  ctx    : CanvasRenderingContext2D,
  alpha  : number,
  color  : string,
  radius : number,
  x      : number,
  y      : number
): void {
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.globalAlpha = alpha
  ctx.fillStyle   = color
  ctx.fill()
  ctx.globalAlpha = 1
  ctx.setLineDash(DASH)
  ctx.lineWidth   = 1.5
  ctx.strokeStyle = color
  ctx.stroke()
  ctx.setLineDash(LINE)
}

/**
 * Paints the front page's sky over the whole drawing: the night at the top
 * easing down through the scheme's bands to the low sky by about two thirds of
 * the way down.
 */
export function sky(ctx: CanvasRenderingContext2D, g: Geometry, tones: Tones): void {
  const tone    = tones.sky(g.scheme)
  const count   = tone.bands.length
  ctx.fillStyle = tone.top
  ctx.fillRect(0, 0, g.w, g.h)
  for (let k = 1; k < count; k++) {
    const top     = g.h * 0.72 * (k - 0.5) / (count - 1)
    ctx.fillStyle = tone.bands[k]!
    ctx.fillRect(0, top, g.w, g.h - top)
  }
}

/**
 * The two strips along the foot of the neighbors drawing, counting over the
 * last seconds the seven by rank, a solid bar, against the birds inside the
 * fixed range, a bar whose shade deepens with its count, on one time axis.
 * Each strip is labeled at its end, or above a strip as wide as the panel
 * where the labels leave the strips too little room. The trace and every word
 * are written once, the count's word once per change of count, so a frame
 * builds nothing.
 */
export class Strips {
  readonly #font  : string
  panel           : Rect = { x0: 0, x1: 0, y0: 0, y1: 0 }
  readonly #tones : Tones
  readonly #trace : Float32Array
  #traced         = 0
  readonly #words : { ago: string, counted: number, range: string, seven: string }

  constructor(font: string, samples: number, seconds: number, tones: Tones) {
    this.#font  = font
    this.#tones = tones
    this.#trace = new Float32Array(samples)
    this.#words = {
      ago     : `${seconds} s ago`,
      counted : -1,
      range   : '',
      seven   : `seven by rank, ${common.NEIGHBORS}`
    }
  }

  /**
   * Reads the newest sample of the trace.
   */
  get last(): number {
    return this.#trace[(this.#traced + this.#trace.length - 1) % this.#trace.length]!
  }

  /**
   * Writes `words` in the code face in `color` over a halo of the sky, so it
   * reads over any bird.
   */
  #label(
    ctx   : CanvasRenderingContext2D,
    align : CanvasTextAlign,
    color : string,
    g     : Geometry,
    words : string,
    x     : number,
    y     : number
  ): void {
    ctx.textAlign   = align
    ctx.lineJoin    = 'round'
    ctx.lineWidth   = 3
    ctx.strokeStyle = this.#tones.sky(g.scheme).top
    ctx.strokeText(words, x, y)
    ctx.fillStyle   = color
    ctx.fillText(words, x, y)
    ctx.textAlign   = 'left'
  }

  /**
   * Draws one strip from `left` to `right`, `tall` pixels from `top`: the
   * seven's solid, and the range's shaded by each sample's count against the
   * seven.
   */
  #strip(
    ctx   : CanvasRenderingContext2D,
    color : string,
    left  : number,
    right : number,
    row   : number,
    tall  : number,
    top   : number
  ): void {
    const count   = this.#trace.length
    const width   = (right - left) / count
    ctx.fillStyle = color
    for (let k = 0; k < count; k++) {
      const value     = this.#trace[(this.#traced + k) % count]!
      ctx.globalAlpha = row === 0 ? 0.85 : 0.15 + 0.7 * common.clamp(value / common.NEIGHBORS)
      ctx.fillRect(left + k * width, top, width + 0.5, tall)
    }
    ctx.globalAlpha = 1
  }

  /**
   * Fills the whole trace with `count`, as though the count had held for the
   * whole of it.
   */
  fill(count: number): void {
    this.#trace.fill(count)
  }

  /**
   * Draws the strips under a band of the sky's night, the seven's in `seven`
   * and the range's in `ranged`.
   */
  paint(ctx: CanvasRenderingContext2D, g: Geometry, ranged: string, seven: string): void {
    const { panel } = this
    const words     = this.#words
    const faint     = this.#tones.ink(g.scheme, 0.55)
    const left      = panel.x0 + 12
    if (words.counted !== this.last) {
      words.counted = this.last
      words.range   = `within range, ${this.last}`
    }
    ctx.fillStyle   = this.#tones.sky(g.scheme).top
    ctx.globalAlpha = 0.82
    ctx.fillRect(panel.x0, panel.y0, panel.x1 - panel.x0, panel.y1 - panel.y0)
    ctx.globalAlpha = 1
    ctx.font        = this.#font
    const widest    = Math.max(ctx.measureText(words.seven).width, ctx.measureText(words.range).width)
    const right     = panel.x1 - 18 - widest
    const stacked   = right - left < (panel.x1 - panel.x0) * 0.55
    const end       = stacked ? panel.x1 - 12 : right
    const top       = panel.y0 + 10
    const tall      = stacked ? 11 : (panel.y1 - 30 - panel.y0) / 2 - 2
    const next      = stacked ? panel.y0 + 49 : top + tall + 4
    const foot      = panel.y1 - (stacked ? 5 : 6)
    this.#label(ctx, 'left', faint, g, words.ago, left, foot)
    this.#label(ctx, 'right', faint, g, 'now', end, foot)
    this.#strip(ctx, seven, left, end, 0, tall, stacked ? panel.y0 + 19 : top)
    this.#strip(ctx, ranged, left, end, 1, tall, next)
    if (stacked) {
      this.#label(ctx, 'left', seven, g, words.seven, left, panel.y0 + 15)
      this.#label(ctx, 'left', ranged, g, words.range, left, panel.y0 + 45)
      return
    }
    this.#label(ctx, 'left', seven, g, words.seven, right + 8, top + tall / 2 + 4)
    this.#label(ctx, 'left', ranged, g, words.range, right + 8, next + tall / 2 + 4)
  }

  /**
   * Takes `count` as the trace's newest sample, dropping its oldest.
   */
  sample(count: number): void {
    this.#trace[this.#traced] = count
    this.#traced              = (this.#traced + 1) % this.#trace.length
  }
}
