import type { Marks }                from '../canvas/marks.ts'
import * as part                     from '../canvas/part.ts'
import type { Tones }                from '../canvas/tones.ts'
import { clamp, mulberry }           from '../common.ts'
import type { Identity }             from '../identity.ts'
import { EasedPointer }              from './eased.ts'
import { Links }                     from './links.ts'
import { Numerals }                  from './numerals.ts'
import { type Oval, passing, rimOf } from './passing.ts'

/**
 * The murmuration streaming across the drawing, one entry per bird in each
 * array: where it flows across, `x`, the line it held far upstream, `far`, its
 * length, its traced bird, and the phase of its sway, and as each frame places
 * it, how far down it flies, its heading, its size, and how warm the parting
 * makes it, beside how warm the links should make it, `glow`, and how warm they
 * have made it so far, `lit`.
 */
interface Birds {
  angle  : Float32Array
  far    : Float32Array
  glow   : Float32Array
  glyph  : Uint8Array
  length : Float32Array
  lit    : Float32Array
  phase  : Float32Array
  size   : Float32Array
  tone   : Float32Array
  x      : Float32Array
  y      : Float32Array
}

/**
 * How the seven nearest show: the width, the opacity, and the tone of their
 * hairlines, how long the pointer rests, in seconds, before they show, how warm
 * and how much longer the picked bird's seven grow, and how fast a bird warms
 * and cools, in rates a second.
 */
const SEVEN = {
  alpha : 0.45,
  fall  : 8,
  grow  : 0.12,
  hue   : 0.55,
  line  : 0.7,
  rest  : 0.4,
  rise  : 14,
  warm  : 0.55
} as const
/**
 * The murmuration: `count` birds over a drawing 1100 pixels wide, scaled with
 * its width from 45% to 110% of that, drawn from `seed`, each flowing `speed`
 * pixels a second, and the pointer picking the bird nearest it within `pick`
 * pixels.
 */
const STREAM = { count: 1500, pick: 90, seed: 404, speed: 24 } as const

/**
 * Builds the arrays of a murmuration of `count` birds.
 */
function flock(count: number): Birds {
  const floats = (): Float32Array => new Float32Array(count)
  return {
    angle  : floats(),
    far    : floats(),
    glow   : floats(),
    glyph  : new Uint8Array(count),
    length : floats(),
    lit    : floats(),
    phase  : floats(),
    size   : floats(),
    tone   : floats(),
    x      : floats(),
    y      : floats()
  }
}

/**
 * The page met at a missing address as a drawing: a murmuration streaming
 * across the page that opens a hole around the 404, the way a flock opens
 * around a falcon. Each bird keeps to its own line through the stream, the
 * lines parting around the numerals and closing again behind them, and the
 * birds at the rim crowd, turn, and warm. A click bursts the hole wider for a
 * moment. Once the pointer rests in the stream, faint hairlines reach out from
 * the bird nearest it to that bird's seven nearest neighbors, one after
 * another, and draw back as soon as the pointer moves, whereas while the
 * drawing stands still they show at once on a resting pointer. Every array is
 * made when the drawing fits its box, so a frame makes nothing new.
 */
export class Opening implements part.Part {
  readonly #aim      = new EasedPointer()
  #birds             = flock(0)
  #burst             = 0
  #calm              = 0
  #chosen            = -1
  #last              = 0
  readonly #links    = new Links()
  readonly #marks    : Marks
  readonly #numerals : Numerals
  readonly #oval     : Oval = { a: 0, b: 0, cx: 0, cy: 0 }
  readonly #rested   = [0, 0]
  readonly #tones    : Tones

  constructor(marks: Marks, tones: Tones, face: Identity['faces']['display']) {
    this.#marks    = marks
    this.#numerals = new Numerals(face)
    this.#tones    = tones
  }

  /**
   * Draws the flock, then the hairlines beneath the seven, then the picked bird
   * and its seven on top, warmer and longer as the links light them, and the
   * numerals last.
   */
  #draw(ctx: CanvasRenderingContext2D, g: part.Geometry, dpr: number, birds: Birds): void {
    const { angle, glyph, lit, size, tone, x, y } = birds
    const marks                                   = this.#marks
    const tones                                   = this.#tones
    const alpha                                   = g.scheme === 'dark' ? 0.78 : 0.7
    for (let i = 0; i < x.length; i++) {
      if (lit[i]! >= 0.01) continue
      const color = tones.flock(g.scheme, tone[i]!, alpha)
      marks.bird(ctx, dpr, glyph[i]!, x[i]!, y[i]!, angle[i]!, size[i]!, color)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.globalAlpha = SEVEN.alpha
    ctx.strokeStyle = tones.flock(g.scheme, SEVEN.hue, 1)
    ctx.lineWidth   = SEVEN.line
    this.#links.paint(ctx, x, y)
    for (let i = 0; i < x.length; i++) {
      const light = lit[i]!
      if (light < 0.01) continue
      const warm      = Math.max(tone[i]!, SEVEN.warm * (0.3 + 0.7 * Math.min(1, light / 0.8)))
      ctx.globalAlpha = alpha + (1 - alpha) * Math.min(1, light / 0.3)
      const length    = size[i]! * (1 + SEVEN.grow * light)
      marks.bird(ctx, dpr, glyph[i]!, x[i]!, y[i]!, angle[i]!, length, tones.flock(g.scheme, warm, 1))
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.globalAlpha = 1
    this.#numerals.paint(ctx, tones.flock(g.scheme, 0, 0.92))
  }

  /**
   * Picks the bird nearest the pointer within reach, keeping the bird picked
   * last while it stays nearly as near, so a resting pointer hands on only as
   * the stream carries the bird away, and none while the pointer is away.
   */
  #pick(birds: Birds, g: part.Geometry): number {
    const { on, x, y } = this.#aim.frame(g)
    const reach        = STREAM.pick ** 2
    const count        = on > 0.02 ? birds.x.length : 0
    let pick           = -1
    let nearest        = reach
    for (let bird = 0; bird < count; bird++) {
      const distance = (birds.x[bird]! - x) ** 2 + (birds.y[bird]! - y) ** 2
      if (distance >= nearest) continue
      nearest = distance
      pick    = bird
    }
    const chosen = this.#chosen
    if (chosen >= 0 && pick >= 0) {
      const kept = (birds.x[chosen]! - x) ** 2 + (birds.y[chosen]! - y) ** 2
      if (kept < reach && kept < nearest * 2.2) pick = chosen
    }
    this.#chosen = pick
    return pick
  }

  /**
   * Places each bird for this frame on the line it keeps around the oval the
   * numerals and any burst open, swaying while motion runs, turned along its
   * line, and warmer and longer the nearer the rim it flies.
   */
  #place(birds: Birds, g: part.Geometry): void {
    const oval  = this.#oval
    const swell = this.#burst > 0 ? 1 + 0.55 * Math.sin(Math.PI * Math.min(1, (1 - this.#burst) * 1.4)) : 1
    const sway  = g.still ? 0 : 1.2

    oval.a  = (this.#numerals.width / 2 + 36) * swell
    oval.b  = (this.#numerals.height / 2 + 34) * swell
    oval.cx = this.#numerals.cx
    oval.cy = this.#numerals.cy
    for (let i = 0; i < birds.x.length; i++) {
      const x        = birds.x[i]!
      const far      = birds.far[i]!
      const y        = passing(x, far, oval) + sway * Math.sin(g.t * 0.6 + birds.phase[i]!)
      const rim      = rimOf(oval, x, y)
      const warm     = Math.min(1, Math.exp(-(rim - 1) * 5))
      birds.y[i]     = y
      birds.angle[i] = Math.atan2(passing(x + 3, far, oval) - passing(x - 3, far, oval), 6)
      birds.size[i]  = birds.length[i]! * (1 + 0.15 * warm)
      birds.tone[i]  = 0.05 + 0.9 * warm
    }
  }

  /**
   * Moves the links a frame on: they reach out once the pointer has rested on a
   * picked bird, at once while the drawing stands still, and each bird warms
   * toward the glow its link gives it and cools once it is let go.
   */
  #settle(birds: Birds, g: part.Geometry, dt: number, pick: number): void {
    const { x, y } = this.#aim
    const rested   = this.#rested
    const moved    = Math.hypot(x - rested[0]!, y - rested[1]!)
    this.#calm     = g.still ? SEVEN.rest : moved < dt * 30 ? this.#calm + dt : 0
    rested[0]      = x
    rested[1]      = y
    const out      = pick >= 0 && g.pointer !== null && this.#calm >= SEVEN.rest
    if (pick >= 0) this.#links.pick(pick, birds.x, birds.y, dt, g.still)
    this.#links.reachOn(dt, out, g.still)
    this.#links.fade(dt, out, g.still)
    const { glow, lit } = birds
    const rise          = 1 - Math.exp(-dt * SEVEN.rise)
    const fall          = 1 - Math.exp(-dt * SEVEN.fall)
    glow.fill(0)
    if (pick >= 0) glow[pick] = this.#links.anchor.w
    this.#links.glow(glow)
    for (let i = 0; i < lit.length; i++) {
      lit[i] = g.still ? glow[i]! : lit[i]! + (glow[i]! - lit[i]!) * (glow[i]! > lit[i]! ? rise : fall)
    }
  }

  /**
   * Lays the murmuration across a drawing `w` by `h` pixels, each bird on a
   * line about a ribbon that wavers across the drawing, and measures the
   * numerals at its middle, letting every link go.
   */
  fit({ h, w }: part.Geometry): void {
    const draw   = mulberry(STREAM.seed)
    const birds  = flock(Math.round(clamp(w / 1100, 0.45, 1.1) * STREAM.count))
    const shrunk = w < part.NARROW ? 3.4 : 3.9
    for (let i = 0; i < birds.x.length; i++) {
      const x         = draw() * w
      const middle    = h * (0.5 + 0.07 * Math.sin(x / w * Math.PI * 2 + 0.6) + 0.03 * Math.sin(x / 61))
      const half      = h * (0.2 + 0.08 * Math.sin(x / w * Math.PI * 3 + 1.4) + 0.04 * Math.sin(x / 37))
      birds.x[i]      = x
      birds.far[i]    = middle + (draw() + draw() + draw() - 1.5) / 1.5 * half
      birds.length[i] = shrunk + 2.8 * draw() ** 1.7
      birds.glyph[i]  = Math.floor(draw() * this.#marks.count)
      birds.phase[i]  = draw() * Math.PI * 2
    }
    this.#birds  = birds
    this.#chosen = -1
    this.#numerals.fit(w, h)
    this.#links.reset()
  }

  /**
   * Places and paints the flock for one frame, moving the links on by the time
   * since the last paint, or at once while the drawing stands still.
   */
  paint(ctx: CanvasRenderingContext2D, g: part.Geometry, dpr: number): void {
    const birds = this.#birds
    const dt    = g.still ? 1 : g.t - this.#last
    this.#last  = g.t
    this.#place(birds, g)
    this.#settle(birds, g, dt, this.#pick(birds, g))
    this.#draw(ctx, g, dpr, birds)
  }

  /**
   * Bursts the hole around the numerals wider for a moment.
   */
  press(): void {
    this.#burst = 1
  }

  /**
   * Flows each bird `dt` seconds on across the drawing, wrapping it back to the
   * left edge once it passes the right, eases the pointer, and lets a burst
   * settle.
   */
  step(dt: number, g: part.Geometry): void {
    this.#burst = Math.max(0, this.#burst - dt * 0.8)
    this.#aim.step(dt, g.pointer)
    const birds = this.#birds
    for (let i = 0; i < birds.x.length; i++) {
      birds.x[i]! += dt * STREAM.speed
      if (birds.x[i]! > g.w + 20) birds.x[i]! -= g.w + 40
    }
  }
}
