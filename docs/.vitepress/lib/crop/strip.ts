import type { Marks }          from '../canvas/marks.ts'
import type { Geometry, Part } from '../canvas/part.ts'
import type { Tones }          from '../canvas/tones.ts'
import { type Box, clamp }     from '../common.ts'
import * as flock              from '../front/flock.ts'
import { type Heat, joined }   from '../front/heat.ts'
import { Plume, RISE }         from '../front/plume.ts'
import { PointerHeat }         from '../front/pointer-heat.ts'
import type { Murmur }         from '../front/sky.ts'
import { type Kept, MOST_M }   from './kept.ts'

/**
 * How far along the flock's gradient its birds are drawn, from its foot at 0 to
 * its head at 1.
 */
const EMBER = 0.85
/**
 * How far inside the part's edge a bird starts to fade, in pixels at most and
 * as a share of the part's height.
 */
const FADE = { most: 5, share: 0.12 } as const
/**
 * How far above the ground line the part's foot stands, in pixels.
 */
const FOOT = 2
/**
 * The opacity of the ground line's ink.
 */
const GROUND = 0.22
/**
 * How long a bird is drawn against the length the flock gives it, before the
 * sky's scale.
 */
const LENGTH = 0.8
/**
 * How opaque a bird is drawn at its fullest.
 */
const OPACITY = 0.82
/**
 * Where the unseen plume stands: its share of the way across the sky, and how
 * far below the sky's foot in the front page's pixels.
 */
const PLUME = { across: 0.76, below: 36 } as const
/**
 * The shortest a bird is drawn, in pixels.
 */
const SHORTEST = 2.2
/**
 * How tall the flock's sky is in its own pixels, whatever the part kept, about
 * the room a body wheeling at the flock's cruise takes to turn.
 */
const SKY_TALL = 120
/**
 * How many birds showing spread across the whole body. Fewer gather about their
 * own middle by the root of their share of this, while the flock flies on
 * unchanged.
 */
const SPREAD = 68
/**
 * The opacity of the part's tint in each scheme.
 */
const TINT = { dark: 0.08, light: 0.09 } as const
/**
 * The height of this sky against the front page's, 484 pixels tall at a window
 * 1,280 pixels wide.
 */
const shrink = SKY_TALL / 484
/**
 * The front page's plume at the scale of this sky.
 */
const shrunk = { ...RISE, core: RISE.core * shrink, reach: RISE.reach * shrink }

/**
 * Reads where `place` stands once what surrounds `middle` gathers by `tight`.
 */
const gathered = (middle: number, place: number, tight: number): number => middle + (place - middle) * tight

/**
 * Reads how much of `bird` shows while `showing` birds show, a bird fading in
 * over the span of three.
 */
const shownOf = (bird: number, showing: number): number => clamp(showing - bird + 1.5, 0, 3) / 3

/**
 * Reads how far down a strip `tall` pixels high `height` meters stand, on a
 * square root of the height above `floor`, from the top of the strip at the
 * most the menus keep to the part's foot at `floor`.
 */
export const stripY = (floor: number, height: number, tall: number): number =>
  (tall - 1 - FOOT) * (1 - Math.sqrt((height - floor) / (MOST_M - floor)))

/**
 * The part the menus keep as a tinted block over a ground line, with the front
 * page's flock flying inside it, every flight value unchanged and through the
 * same air, a plume that draws nothing here and stands for nothing in the
 * sample. The flock flies a sky `SKY_TALL` tall and as wide as the part at the
 * same scale on both axes, its birds showing in step with the share the menus
 * keep. Each glide of the part reaches the flock through `Flock.region`, the
 * sky swapped between two records since `region` keeps the record it receives
 * and compares the next against it.
 */
export class FlightStrip implements Part {
  readonly #drawn  = { tight: 1, x: 0, y: 0 }
  readonly #falcon = new PointerHeat()
  readonly #field  : Heat
  readonly #flock  : flock.Flock
  readonly #kept   : Kept
  readonly #marks  : Marks
  readonly #plume  = new Plume(shrunk)
  #settle          : number
  #sky             : Box = { x0: 0, x1: 0, y0: 0, y1: SKY_TALL }
  #spare           : Box = { x0: 0, x1: 0, y0: 0, y1: SKY_TALL }
  readonly #tones  : Tones

  constructor(kept: Kept, marks: Marks, tones: Tones, fire: number, murmur: Murmur) {
    this.#kept       = kept
    this.#marks      = marks
    this.#tones      = tones
    this.#field      = joined(this.#plume, this.#falcon)
    this.#flock      = new flock.Flock({
      ...flock.FLIGHT,
      count  : murmur.count,
      glyphs : marks.count,
      roost  : flock.roost
    })
    this.#plume.fire = fire
    this.#plume.y    = SKY_TALL + PLUME.below * shrink
    this.#settle     = murmur.settle
  }

  /**
   * Reads where the birds showing stand and how close they gather, for this
   * frame's drawing and the pointer's reading.
   */
  #gather(showing: number): void {
    const birds = this.#flock
    const drawn = this.#drawn
    let weight  = 0, x = 0, y = 0
    for (let bird = 0; bird < birds.x.length; bird++) {
      const shown = shownOf(bird, showing)
      if (shown <= 0) break
      weight += shown
      x      += birds.x[bird]! * shown
      y      += birds.y[bird]! * shown
    }
    drawn.tight = Math.min(1, Math.sqrt(showing / SPREAD))
    drawn.x     = x / weight
    drawn.y     = y / weight
  }

  /**
   * Widens or narrows the sky to the part kept where its width changed.
   */
  #reach(g: Geometry): void {
    const { east, west } = this.#kept
    const wide           = (east.value - west.value) * g.w / this.#scale(g)
    if (Math.abs(wide - this.#sky.x1) < 0.01) return
    const next    = this.#spare
    next.x1       = wide
    this.#spare   = this.#sky
    this.#sky     = next
    this.#plume.x = wide * PLUME.across
    this.#flock.region(next)
  }

  /**
   * Reads how many strip pixels one pixel of the sky stands for.
   */
  #scale(g: Geometry): number {
    const { model, top } = this.#kept
    return (this.#y(model.floor_m, g) - this.#y(top.value, g)) / SKY_TALL
  }

  /**
   * Reads how far down this strip `height` meters stand, in pixels.
   */
  #y(height: number, g: Geometry): number {
    return stripY(this.#kept.model.floor_m, height, g.h)
  }

  /**
   * Fits the sky to the part kept, settling the flock on the first fit alone.
   */
  fit(g: Geometry): void {
    this.#reach(g)
    this.#flock.settle(this.#settle, this.#field)
    this.#settle = 0
  }

  /**
   * Paints the part's tint and the ground line, then each bird showing at its
   * gathered place, fading out as it nears the part's edge and drawing nothing
   * outside it. A drawing standing still paints the part at its target.
   */
  paint(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number): void {
    const kept = this.#kept
    if (g.still) {
      kept.settle()
      this.#reach(g)
    }
    const birds   = this.#flock
    const drawn   = this.#drawn
    const west    = kept.west.value * g.w
    const east    = kept.east.value * g.w
    const top     = this.#y(kept.top.value, g)
    const foot    = this.#y(kept.model.floor_m, g)
    const scale   = this.#scale(g)
    const fade    = Math.min(FADE.most, (foot - top) * FADE.share)
    const length  = LENGTH * Math.sqrt(scale)
    const showing = kept.share * birds.x.length
    const tone    = this.#tones.flock(g.scheme, EMBER, 1)
    ctx.fillStyle = this.#tones.flock(g.scheme, 1, TINT[g.scheme])
    ctx.fillRect(west, top, east - west, foot - top)
    ctx.fillStyle = this.#tones.ink(g.scheme, GROUND)
    ctx.fillRect(0, g.h - 1, g.w, 1)
    this.#gather(showing)
    for (let bird = 0; bird < birds.x.length; bird++) {
      const shown = shownOf(bird, showing)
      if (shown <= 0) break
      const x    = west + gathered(drawn.x, birds.x[bird]!, drawn.tight) * scale
      const y    = top + gathered(drawn.y, birds.y[bird]!, drawn.tight) * scale
      const room = Math.min(x - west, east - x, y - top, foot - y)
      if (room <= 0) continue
      const angle     = Math.atan2(birds.vy[bird]!, birds.vx[bird]!)
      const long      = Math.max(birds.size[bird]! * length, SHORTEST)
      ctx.globalAlpha = shown * OPACITY * clamp(room / fade)
      this.#marks.bird(ctx, dpr, birds.glyph[bird]!, x, y, angle, long, tone)
    }
    ctx.globalAlpha = 1
  }

  /**
   * Moves the part's glides, the sky, the plume, and the flock on, and puts the
   * falcon where the pointer reads through the gathering, so a bird parts from
   * the pointer at the place it is drawn.
   */
  step(dt: number, g: Geometry): void {
    const kept = this.#kept
    kept.step(dt)
    this.#reach(g)
    const drawn     = this.#drawn
    const scale     = this.#scale(g)
    const across    = ((g.pointer?.[0] ?? 0) - kept.west.value * g.w) / scale
    const down      = ((g.pointer?.[1] ?? 0) - this.#y(kept.top.value, g)) / scale
    this.#falcon.on = g.pointer !== null
    this.#falcon.x  = gathered(drawn.x, across, 1 / drawn.tight)
    this.#falcon.y  = gathered(drawn.y, down, 1 / drawn.tight)
    this.#plume.t  += dt
    this.#flock.step(dt, this.#field)
  }
}
