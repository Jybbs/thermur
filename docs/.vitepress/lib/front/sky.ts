import type { Marks }  from '../canvas/marks.ts'
import * as part       from '../canvas/part.ts'
import type { Tones }  from '../canvas/tones.ts'
import * as common     from '../common.ts'
import type { Scheme } from '../identity.ts'
import { Flock }       from './flock.ts'
import type { Heat }   from './heat.ts'
import { Plume }       from './plume.ts'
import { PointerHeat } from './pointer-heat.ts'

/**
 * The flock the sky draws: how many birds it holds and how many seconds it
 * flies unseen before the first frame, so the page opens on a formed flock.
 */
export interface Murmur {
  count  : number
  settle : number
}

/**
 * How the sky's flock flies: from one seed, each bird between the two lengths
 * `size` holds, keeping `sep` pixels from its nearest neighbors, and cruising
 * at `speed` pixels a second.
 */
const FLIGHT = { seed: 41, sep: 9, size: [4, 6.6], speed: 30 } as const
/**
 * The waves of the ridge at the hero's foot, which cancel at the base the sky's
 * fit sets 14 pixels below the horizon.
 */
const RIDGE = { amplitude: 14, periods: [150, 61, 23], phases: [1.1, 2.2, 3.3] } as const

/**
 * Wanders the flock's roost from one side of the sky to the other, writing
 * where it stands at `t` inside `box` into `out`.
 */
function roost(t: number, box: common.Box, out: [number, number]): void {
  const side = Math.sin(t * 0.035 + 2.2) >= 0 ? 1 : -1
  out[0]     = box.x0 + (box.x1 - box.x0) * (0.5 + side * (0.3 + 0.08 * Math.sin(t * 0.11)))
  out[1]     = box.y0 + (box.y1 - box.y0) * (0.5 + 0.22 * Math.sin(t * 0.09))
}

/**
 * Builds the ground beneath `height` across a drawing `w` pixels wide as a
 * path, traced every `step` pixels, six unless given, and closed along
 * `bottom`.
 */
export function groundPath(bottom: number, height: (x: number) => number, w: number, step = 6): Path2D {
  const path = new Path2D()
  path.moveTo(0, height(0))
  for (let x = step; x <= w + step; x += step) path.lineTo(x, height(x))
  path.lineTo(w + step, bottom)
  path.lineTo(0, bottom)
  path.closePath()
  return path
}

/**
 * Measures how much of a drawing stays at a point in the window, falling away
 * by up to `depth` in an oval behind the hero's wordmark, tagline, and button
 * and by 0.7 over the reading column below it, so the words stand on a clear
 * ground.
 */
export function quietAt(g: part.Geometry, x: number, y: number, depth: number): number {
  const rx   = Math.min(g.w * 0.44, 540)
  const ry   = Math.max(130, g.hero.h * 0.34)
  const cy   = g.hero.top + g.hero.h * 0.5 - g.dy
  const oval = Math.exp(-(((x - g.w / 2) / rx) ** 2 + ((y - cy) / ry) ** 2) * 1.4)
  const d    = g.doc
  if (d === null) return 1 - depth * oval
  const top    = d.top - g.dy
  const bottom = d.bottom - g.dy
  const across = common.smoothstep(d.left - 40, d.left + 10, x)
               * (1 - common.smoothstep(d.right - 10, d.right + 40, x))
  const down = common.smoothstep(top - 40, top, y) * (1 - common.smoothstep(bottom, bottom + 40, y))
  return (1 - depth * oval) * (1 - 0.7 * across * down)
}

/**
 * The hero's sky at dusk, banded from the night above down to a ridge at the
 * hero's foot, a short fire on the ridge lifting the bands with its plume, and
 * the flock a silhouette crossing the low bright sky, parting where the plume
 * rises and from the reader's pointer, which acts as the falcon. The light
 * scheme takes the dawn.
 */
export class Sky implements part.Part {
  horizon             = 0
  #bands              : readonly Path2D[] = []
  #edge               : Path2D | null = null
  readonly #falcon    = new PointerHeat(80)
  readonly #fire      = new Plume({ core: 22, gain: 1.7, lean: 0.42, reach: 300, spread: 0.24 })
  readonly #field     : Heat = { heat: (x, y) => this.#fire.heat(x, y) + this.#falcon.heat(x, y) }
  readonly #flock     : Flock
  readonly #gradients = new Map<Scheme, CanvasGradient>()
  readonly #mark      = { fire: 0, half: 26, scheme: 'light' as Scheme, x: 0, y: 0 }
  readonly #marks     : Marks
  readonly #ridge     = { ...RIDGE, base: 0 }
  #settle             : number
  readonly #tones     : Tones

  constructor(marks: Marks, tones: Tones, fire: number, murmur: Murmur) {
    this.#marks     = marks
    this.#tones     = tones
    this.#fire.fire = fire
    this.#mark.fire = fire
    this.#settle    = murmur.settle
    this.#flock     = new Flock({ ...FLIGHT, count: murmur.count, glyphs: marks.count, roost })
  }

  /**
   * Strokes the fire's front along the ridge, a hairline 52 pixels long in
   * Kindleflame, at the fire's strength.
   */
  #front(ctx: CanvasRenderingContext2D, g: part.Geometry): void {
    ctx.save()
    ctx.globalAlpha = 0.9 * this.#fire.fire
    ctx.strokeStyle = this.#tones.heat[0]!
    ctx.lineWidth   = 1.8
    ctx.lineCap     = 'round'
    ctx.beginPath()
    ctx.moveTo(this.#fire.x - 26, this.ridge(this.#fire.x - 26) - g.dy - 1)
    for (let x = this.#fire.x - 24; x <= this.#fire.x + 26; x += 2) ctx.lineTo(x, this.ridge(x) - g.dy - 1)
    ctx.stroke()
    ctx.restore()
  }

  /**
   * Reads the ridge's gradient in `scheme`, solid for the scheme's solid height
   * and fading out below it, built once per scheme.
   */
  #gradient(ctx: CanvasRenderingContext2D, scheme: Scheme): CanvasGradient {
    return common.cached(this.#gradients, scheme, () => {
      const sky      = this.#tones.sky(scheme)
      const gradient = ctx.createLinearGradient(0, 0, 0, sky.solid + sky.fade)
      gradient.addColorStop(0, sky.ridge)
      if (sky.solid > 0) gradient.addColorStop(sky.solid / (sky.solid + sky.fade), sky.ridge)
      else gradient.addColorStop(0.35, sky.ridgeSoft)
      gradient.addColorStop(1, sky.ridgeClear)
      return gradient
    })
  }

  /**
   * Measures how far the plume raises a band's edge at a point, up to 16 pixels
   * at the plume's center, scaled by the fire's strength.
   */
  #lift(x: number, y: number): number {
    const rise   = this.horizon - y
    const { lean, spread } = this.#fire.rise
    const center = this.#fire.x + lean * rise
    const width  = 30 + spread * rise
    const fade   = 0.3 + 0.7 * Math.sin(Math.PI * Math.min(1, rise / Math.max(1, this.horizon) + 0.1))
    return 16 * this.#fire.fire * Math.exp(-((x - center) ** 2) / (2 * width * width)) * fade
  }

  /**
   * Paints the night down to 40 pixels past the horizon, the bands lifting over
   * the plume, and the flock fading behind the hero's words.
   */
  #night(ctx: CanvasRenderingContext2D, g: part.Geometry, dpr: number): void {
    const sky     = this.#tones.sky(g.scheme)
    const top     = -g.dy
    const bottom  = this.horizon - g.dy
    ctx.fillStyle = sky.top
    ctx.fillRect(0, top, g.w, bottom - top + 40)
    ctx.save()
    ctx.translate(0, top)
    for (let k = 0; k < this.#bands.length; k++) {
      ctx.fillStyle = sky.bands[k + 1]!
      ctx.fill(this.#bands[k]!)
    }
    ctx.restore()
    const flock = this.#flock
    const span  = Math.max(1, this.horizon - g.hero.top)
    for (let i = 0; i < flock.x.length; i++) {
      const x         = flock.x[i]!
      const y         = flock.y[i]! - g.dy
      const depth     = common.clamp((flock.y[i]! - g.hero.top) / span)
      const angle     = Math.atan2(flock.vy[i]!, flock.vx[i]!)
      const shade     = sky.shadeAlpha[0] + (sky.shadeAlpha[1] - sky.shadeAlpha[0]) * depth
      ctx.globalAlpha = shade * quietAt(g, x, y, 0.92)
      this.#marks.bird(ctx, dpr, flock.glyph[i]!, x, y, angle, flock.size[i]!, sky.shade)
    }
    ctx.globalAlpha = 1
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  /**
   * Sets the horizon at the hero's foot, the fire on the ridge, the flock's
   * region in the sky above it, and the outlines of the bands lifting over the
   * plume, in the page's own coordinates, and of the ridge, traced down from
   * the horizon, settling the flock on the first fit alone.
   */
  fit(g: part.Geometry): void {
    this.horizon     = g.hero.bottom - 36
    this.#ridge.base = this.horizon + 14
    this.#fire.x     = g.w * (g.w < part.NARROW ? 0.72 : 0.76)
    this.#fire.y     = this.ridge(this.#fire.x) - 2
    this.#flock.region({ x0: 0, x1: g.w, y0: g.hero.top + 80, y1: this.horizon - 24 })
    const count = this.#tones.sky(g.scheme).bands.length
    this.#edge  = groundPath(320, x => this.ridge(x) - this.horizon, g.w)
    this.#bands = Array.from({ length: count - 1 }, (_, index) => {
      const y = this.horizon * (index + 0.85) / count
      return groundPath(this.horizon + 40, x => y - this.#lift(x, y), g.w, 8)
    })
    this.#flock.settle(this.#settle, this.#field)
    this.#settle = 0
  }

  /**
   * Paints the night, the bands lifting over the plume, and the flock fading
   * behind the hero's words while the sky is in the window, and the fire and
   * the ridge with the fire's line along it until the ridge's fade has scrolled
   * out of the window as well.
   */
  paint(ctx: CanvasRenderingContext2D, g: part.Geometry, dpr: number): void {
    const bottom = this.horizon - g.dy
    if (bottom < -320) return
    if (bottom >= -40) this.#night(ctx, g, dpr)
    this.#mark.scheme = g.scheme
    this.#mark.x      = this.#fire.x
    this.#mark.y      = this.#fire.y - g.dy
    this.#marks.fire(ctx, this.#mark)
    ctx.save()
    ctx.translate(0, bottom)
    ctx.fillStyle = this.#gradient(ctx, g.scheme)
    ctx.fill(this.#edge!)
    ctx.restore()
    this.#front(ctx, g)
  }

  /**
   * Reads the height of the ridge at the hero's foot `x` pixels across.
   */
  ridge(x: number): number {
    return common.heightAt(this.#ridge, x)
  }

  /**
   * Sways the plume, puts the falcon at the pointer while it is over the sky,
   * and moves the flock on.
   */
  step(dt: number, g: part.Geometry): void {
    this.#fire.t   += dt
    this.#falcon.on = g.pointer !== null && g.pointer[1] < this.horizon
    this.#falcon.x  = g.pointer?.[0] ?? 0
    this.#falcon.y  = g.pointer?.[1] ?? 0
    this.#flock.step(dt, this.#field)
  }
}
