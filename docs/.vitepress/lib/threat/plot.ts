import type { Flame, Marks }                  from '../canvas/marks.ts'
import type { Geometry, Part }                from '../canvas/part.ts'
import type { Tones }                         from '../canvas/tones.ts'
import { cached, smoothstep }                 from '../common.ts'
import { type Identity, type Scheme, fontStack } from '../identity.ts'
import { Axes }                               from './axes.ts'
import type { Front }                         from './front.ts'
import type { Playback }                      from './playback.ts'
import { Readings, type Readout }             from './readings.ts'
import { STEPS, type Shades, shadesOf }       from './shades.ts'

/**
 * The dash of the line standing at the front, in pixels.
 */
const DASH = [3, 4]
/**
 * The unbroken line the plot returns to from `DASH`.
 */
const LINE: readonly number[] = []
/**
 * The seconds of the threat's clock a cell takes to light once the front
 * reaches its center, and the fire's mark takes to grow from the western edge.
 */
const WARM = 6

/**
 * The air along the ground as a front of hot air passes through it, on a plot
 * of temperature against distance over a rail a reader can move: the curve of
 * the model's air in the heat ramp, a fading trace at each snapshot it has
 * passed, the cells burning behind the front under the fire's mark, a mark on
 * each cell center warming in turn, and the rail with one tick per snapshot.
 * The curve is written into arrays made when the plot fits, and each path and
 * gradient is built once a fit and each string once, so a frame builds nothing.
 */
export class Plot implements Part {
  readonly #axes     : Axes
  readonly #flame    : Flame = { fire: 0, glow: 64, half: 12, scheme: 'dark', x: 0, y: 0 }
  readonly #front    : Front
  readonly #identity : Identity
  readonly #marks    : Marks
  readonly #pen      : CanvasRenderingContext2D
  readonly #playback : Playback
  readonly #readings : Readings
  readonly #rail     : HTMLElement
  readonly #shades   = new Map<Scheme, Shades>()
  readonly #tick     : string
  readonly #tones    : Tones
  readonly #word     : string

  constructor(
    front    : Front,
    identity : Identity,
    marks    : Marks,
    playback : Playback,
    readout  : Readout,
    tones    : Tones
  ) {
    this.#axes     = new Axes(front)
    this.#front    = front
    this.#identity = identity
    this.#marks    = marks
    this.#playback = playback
    this.#rail     = readout.rail
    this.#readings = new Readings(front, readout)
    this.#tones    = tones
    this.#pen      = document.createElement('canvas').getContext('2d')!
    this.#tick     = `400 11px ${fontStack(identity.faces.mono)}`
    this.#word     = `400 14px ${fontStack(identity.faces.body)}`
  }

  /**
   * Draws the air's curve and the area beneath it, over the last run's where
   * that is still fading.
   */
  #air(ctx: CanvasRenderingContext2D, fading: number, shades: Shades): void {
    const { finale, traces, xs, y1, ys } = this.#axes
    ctx.fillStyle   = shades.area
    ctx.strokeStyle = shades.stroke
    ctx.lineWidth   = 2.25
    if (fading > 0) {
      ctx.globalAlpha = fading
      ctx.fill(finale!)
      ctx.stroke(traces.at(-1)!)
    }
    ctx.beginPath()
    ctx.moveTo(xs[0]!, ys[0]!)
    for (let index = 1; index < xs.length; index++) ctx.lineTo(xs[index]!, ys[index]!)
    ctx.globalAlpha = 0.18
    ctx.lineWidth   = 8
    ctx.stroke()
    ctx.globalAlpha = 1
    ctx.lineWidth   = 2.25
    ctx.stroke()
    ctx.lineTo(xs.at(-1)!, y1)
    ctx.lineTo(xs[0]!, y1)
    ctx.closePath()
    ctx.fill()
  }

  /**
   * Draws a mark on each cell center along the curve, its color reading the
   * air's temperature there and its size swelling as the front's edge passes.
   */
  #cells(ctx: CanvasRenderingContext2D, seconds: number, shades: Shades): void {
    const axes       = this.#axes
    const { lowest } = this.#front
    ctx.lineWidth    = 2
    ctx.strokeStyle  = shades.ground
    for (let column = 0; column < axes.columns; column++) {
      const x       = this.#front.center(column)
      const share   = this.#front.profile(seconds, x)
      const y       = axes.down(this.#front.air(lowest, seconds, x))
      ctx.fillStyle = shades.cells[Math.round(share * STEPS)]!
      ctx.beginPath()
      ctx.arc(axes.across(x), y, 3.4 + 9.6 * share * (1 - share), 0, Math.PI * 2)
      ctx.stroke()
      ctx.fill()
    }
  }

  /**
   * Draws the fire on the ground: each cell whose center the front has reached,
   * lit and flickering, those of the last run fading out with it, and the
   * fire's mark at the front under the line standing there and its name.
   */
  #fire(ctx: CanvasRenderingContext2D, fading: number, g: Geometry, seconds: number, shades: Shades): void {
    const axes              = this.#axes
    const { spacing, span } = this.#front
    const reach             = this.#front.at(seconds)
    const final             = this.#front.at(span)
    const x                 = axes.across(reach)
    ctx.fillStyle           = shades.ember
    for (let column = 0; column < axes.columns; column++) {
      const center = this.#front.center(column)
      const lit    = Math.max(smoothstep(0, WARM, reach - center), center <= final ? fading : 0)
      if (lit === 0) continue
      const from      = axes.across(center - spacing / 2)
      ctx.globalAlpha = lit * (0.74 + 0.2 * Math.sin(g.t * 6.1 + column * 2.4))
      ctx.fillRect(from, axes.y1 - 1.5, Math.min(axes.x1, axes.across(center + spacing / 2)) - from, 3)
    }
    const flicker      = 0.62 + 0.14 * Math.sin(g.t * 9.1) + 0.08 * Math.sin(g.t * 23.7)
    this.#flame.fire   = smoothstep(0, WARM, seconds) * flicker
    this.#flame.scheme = g.scheme
    this.#flame.x      = x
    this.#flame.y      = axes.y1
    this.#marks.fire(ctx, this.#flame)
    ctx.globalAlpha = 0.6
    ctx.strokeStyle = shades.word
    ctx.lineWidth   = 1
    ctx.setLineDash(DASH)
    ctx.beginPath()
    ctx.moveTo(x, axes.y0)
    ctx.lineTo(x, axes.y1)
    ctx.stroke()
    ctx.setLineDash(LINE)
    ctx.globalAlpha  = 1
    ctx.fillStyle    = shades.word
    ctx.font         = this.#word
    ctx.textAlign    = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.fillText('Front', x + 6, axes.y0 + 12)
  }

  /**
   * Draws what stands still: the rules, the temperatures and distances, and the
   * name of each axis.
   */
  #frame(ctx: CanvasRenderingContext2D, shades: Shades): void {
    const axes       = this.#axes
    ctx.strokeStyle  = shades.rule
    ctx.lineWidth    = 1
    ctx.stroke(axes.grid!)
    ctx.fillStyle    = shades.word
    ctx.font         = this.#word
    ctx.textAlign    = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.fillText(axes.height, 0, 14)
    ctx.textAlign    = 'center'
    ctx.fillText(axes.name, (axes.x0 + axes.x1) / 2, axes.y1 + 34)
    ctx.font         = this.#tick
    for (const { value, word } of axes.meters) ctx.fillText(word, axes.across(value), axes.y1 + 15)
    ctx.textAlign    = 'right'
    ctx.textBaseline = 'middle'
    for (const { value, word } of axes.kelvins) ctx.fillText(word, axes.x0 - 7, axes.down(value))
  }

  /**
   * Draws the rail: its track and its snapshots' ticks, the run so far, and the
   * grip at the time drawn, swelling under a press.
   */
  #railway(ctx: CanvasRenderingContext2D, seconds: number, shades: Shades): void {
    const { railY, track, x0, x1 } = this.#axes
    const { grip }  = this.#playback
    const x         = x0 + seconds / this.#front.span * (x1 - x0)
    ctx.strokeStyle = shades.ink
    ctx.globalAlpha = 0.34
    ctx.lineWidth   = 1.5
    ctx.stroke(track!)
    ctx.globalAlpha = 1
    ctx.strokeStyle = shades.lead
    ctx.lineWidth   = 2.5
    ctx.lineCap     = 'round'
    ctx.beginPath()
    ctx.moveTo(x0, railY)
    ctx.lineTo(x, railY)
    ctx.stroke()
    ctx.lineCap     = 'butt'
    ctx.fillStyle   = shades.lead
    ctx.globalAlpha = 0.22 * grip
    ctx.beginPath()
    ctx.arc(x, railY, 13, 0, Math.PI * 2)
    ctx.fill()
    ctx.globalAlpha = 1
    ctx.beginPath()
    ctx.arc(x, railY, 6 + 1.5 * grip, 0, Math.PI * 2)
    ctx.fill()
  }

  /**
   * Draws the trace of each snapshot the front has passed, fading as it ages,
   * and those of the last run fading out together.
   */
  #snapshots(ctx: CanvasRenderingContext2D, fading: number, seconds: number, shades: Shades): void {
    const { interval, span } = this.#front
    const { traces }         = this.#axes
    ctx.strokeStyle          = shades.ink
    ctx.lineWidth            = 1
    for (let index = 0; index < traces.length; index++) {
      const age   = (seconds - index * interval) / span
      const alpha = Math.max(age >= 0 ? 0.42 - 0.28 * age : 0, 0.14 * fading)
      if (alpha === 0) continue
      ctx.globalAlpha = alpha
      ctx.stroke(traces[index]!)
    }
    ctx.globalAlpha = 1
  }

  /**
   * Fits the plot to the canvas: the axes, measured in the words' face, the
   * colors built again on their next read, and the rail's control lined up with
   * the rail drawn.
   */
  fit(g: Geometry): void {
    this.#pen.font = this.#word
    this.#axes.fit(g, this.#pen)
    this.#shades.clear()
    this.#rail.style.setProperty('--threat-rail-left', `${this.#axes.left}px`)
    this.#rail.style.setProperty('--threat-rail-right', `${this.#axes.right}px`)
  }

  /**
   * Draws the plot at the time the clock reads, taking a time a reader set at
   * once where the drawing stands still.
   */
  paint(ctx: CanvasRenderingContext2D, g: Geometry): void {
    if (g.still) this.#playback.settle()
    const { fading, seconds, target } = this.#playback
    const axes   = this.#axes
    const shades = cached(this.#shades, g.scheme, () => shadesOf(
      ctx,
      axes.down(axes.low + 4),
      this.#identity,
      g.scheme,
      this.#tones,
      axes.down(axes.high - 2),
      axes.y1
    ))
    axes.write(seconds)
    this.#frame(ctx, shades)
    this.#snapshots(ctx, fading, seconds, shades)
    this.#air(ctx, fading, shades)
    this.#fire(ctx, fading, g, seconds, shades)
    this.#cells(ctx, seconds, shades)
    this.#railway(ctx, seconds, shades)
    this.#readings.write(seconds, target)
  }

  /**
   * Runs the clock on by one frame.
   */
  step(dt: number): void {
    this.#playback.step(dt)
  }
}
