import type { Marks }           from '../canvas/marks.ts'
import type { Geometry, Part }  from '../canvas/part.ts'
import type { Tones }           from '../canvas/tones.ts'
import { cached, clamp }        from '../common.ts'
import type { Scheme }          from '../identity.ts'
import * as flight              from './flight.ts'
import { type Sky, groundPath } from './sky.ts'

/**
 * What one fit plans for the page as it stands: the flight, the hero's ridge
 * the lines are clipped below, and the ground they land on, both outlined in
 * the page's own coordinates.
 */
interface Fitted {
  clip   : Path2D
  ground : Path2D
  run    : flight.Run
}

/**
 * One drone and its seven nearest neighbors drawn as eight hairlines running
 * down the page beneath the sky, the pen following the reader's scroll on a
 * spring. The lines start from behind the hero's ridge, go quiet over the
 * reading column, part around a fire partway down, and land on a ridge at the
 * page's foot under dusk bands, each line ending in a traced bird warmed by the
 * heat it felt.
 */
export class Lines implements Part {
  readonly #column    = { h: 0, w: 0, x: 0, y: 0 }
  readonly #fades     = new Map<Scheme, CanvasGradient>()
  readonly #fire      : number
  #fitted             : Fitted | null = null
  readonly #mark      = { fire: 0, half: 36, scheme: 'light' as Scheme, x: 0, y: 0 }
  readonly #marks     : Marks
  #pace               = 0
  #shown              = 0
  readonly #sky       : Sky
  readonly #tones     : Tones

  constructor(sky: Sky, marks: Marks, tones: Tones, fire: number) {
    this.#sky       = sky
    this.#marks     = marks
    this.#tones     = tones
    this.#fire      = fire
    this.#mark.fire = fire
  }

  /**
   * Draws the birds at the pen's sample, each turned along its line and warmed
   * by the most heat it felt over the last three seconds, bobbing while motion
   * runs and level once landed.
   */
  #heads(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number, run: flight.Run): void {
    const { hs, xs, ys } = run
    const k              = clamp(Math.round(this.#shown), 1, xs[0]!.length - 1)
    const landed         = k >= xs[0]!.length - 2
    const bob            = g.still ? 0 : 1
    const back           = Math.max(0, k - 6)
    for (let i = 0; i < flight.EIGHT; i++) {
      const angle = landed ? 0 : Math.atan2(ys[i]![k]! - ys[i]![back]!, xs[i]![k]! - xs[i]![back]!)
      let felt    = 0
      for (let q = Math.max(0, k - flight.samplesPerSecond * 3); q <= k; q += 2) {
        felt = Math.max(felt, hs[i]![q]!)
      }
      const hx   = xs[i]![k]! + bob * 1.2 * Math.sin(g.t * 1.3 + i * 1.7)
      const hy   = ys[i]![k]! - g.dy + (landed ? 0 : bob * 1.4 * Math.cos(g.t * 1.1 + i * 2.3))
      const tone = this.#tones.flock(g.scheme, 0.14 + 0.86 * Math.min(1, felt * 2.6), 0.92)
      this.#marks.bird(ctx, dpr, (i * 5 + 3) % this.#marks.count, hx, hy, angle, 9, tone)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  /**
   * Paints the roost at the page's foot, its dusk bands under a fade from the
   * page and the ridge the eight land on, once it nears the window.
   */
  #roost(ctx: CanvasRenderingContext2D, g: Geometry, { ground, run }: Fitted): void {
    const base = run.base - g.dy
    if (base - 120 >= g.h + 40) return
    const sky = this.#tones.sky(g.scheme)
    for (let k = 0; k < sky.roost.bands.length; k++) {
      ctx.fillStyle = sky.roost.bands[k]!
      ctx.fillRect(0, base - 110 + k * 26, g.w, 140 - k * 26)
    }
    const fade = cached(this.#fades, g.scheme, () => {
      const gradient = ctx.createLinearGradient(0, -30, 0, 0)
      gradient.addColorStop(0, sky.top)
      gradient.addColorStop(1, sky.roost.clear)
      return gradient
    })
    ctx.save()
    ctx.translate(0, base - 100)
    ctx.fillStyle = fade
    ctx.fillRect(0, -31, g.w, 32)
    ctx.restore()
    ctx.save()
    ctx.translate(0, -g.dy)
    ctx.fillStyle = sky.ridge
    ctx.fill(ground)
    ctx.restore()
  }

  /**
   * Strokes the samples `from` to `to` of every line at `alpha`, in two passes
   * where the page has a reading column, the line inside the column at less
   * than half the opacity it has outside.
   */
  #stroke(
    ctx   : CanvasRenderingContext2D,
    g     : Geometry,
    run   : flight.Run,
    from  : number,
    to    : number,
    alpha : number
  ): void {
    if (to <= from) return
    const column = this.#column
    const passes = g.doc === null ? 1 : 2
    for (let pass = 0; pass < passes; pass++) {
      const inside = pass === 1
      ctx.save()
      if (g.doc !== null) {
        ctx.beginPath()
        if (!inside) ctx.rect(0, -10, g.w, g.h + 20)
        ctx.rect(column.x, column.y, column.w, column.h)
        ctx.clip(inside ? 'nonzero' : 'evenodd')
      }
      ctx.strokeStyle = this.#tones.ink(g.scheme, inside ? alpha * 0.45 : alpha)
      for (let i = 0; i < flight.EIGHT; i++) {
        ctx.beginPath()
        let pen = false, px = 0, py = 0
        for (let k = from; k <= to; k++) {
          const x = run.xs[i]![k]!
          const y = run.ys[i]![k]! - g.dy
          if (y < -60 || y > g.h + 60) pen = false
          else if (pen) ctx.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2)
          else {
            ctx.moveTo(x, y)
            pen = true
          }
          px = x
          py = y
        }
        ctx.stroke()
      }
      ctx.restore()
    }
  }

  /**
   * Finds the sample the pen draws to with the page where it is now, from the
   * flight's start at the top of the page to its landing at the foot, reaching
   * a few samples ahead at the very top.
   */
  #target(g: Geometry, run: flight.Run): number {
    const pen = flight.penAt(g, g.dy)
    const end = flight.penAt(g, Math.max(0, g.page - g.h))
    return clamp((pen - run.y0) / Math.max(1, end - run.y0)) * (run.xs[0]!.length - 1)
         + 6 * (1 - clamp((pen - run.y0) / 200))
  }

  /**
   * Strokes the lines up to sample `last`, clipped below the hero's ridge, the
   * older stretch faint, the last nine seconds in four darkening steps, and the
   * stretches where a bird felt the heat in the fire's color.
   */
  #trails(ctx: CanvasRenderingContext2D, g: Geometry, { clip, run }: Fitted, last: number): void {
    ctx.save()
    ctx.translate(0, -g.dy)
    ctx.clip(clip)
    ctx.translate(0, g.dy)
    ctx.lineWidth = 1
    ctx.lineJoin  = 'round'
    const dark    = g.scheme === 'dark'
    const fresh   = Math.max(0, last - flight.samplesPerSecond * 9)
    const width   = Math.max(1, Math.floor((last - fresh) / 4))
    this.#stroke(ctx, g, run, 0, fresh, dark ? 0.09 : 0.1)
    for (let s = 0; s < 4; s++) {
      const end = Math.min(last, fresh + (s + 1) * width + 1)
      this.#stroke(ctx, g, run, fresh + s * width, end, (dark ? 0.11 : 0.12) + s * 0.035)
    }
    ctx.strokeStyle = this.#tones.flock(g.scheme, 1, 0.45)
    for (let i = 0; i < flight.EIGHT; i++) {
      ctx.beginPath()
      let pen = false
      for (let k = 1; k <= last; k++) {
        const y = run.ys[i]![k]! - g.dy
        if (run.hs[i]![k]! < 0.03 || y < -60 || y > g.h + 60) pen = false
        else {
          if (!pen) ctx.moveTo(run.xs[i]![k - 1]!, run.ys[i]![k - 1]! - g.dy)
          ctx.lineTo(run.xs[i]![k]!, y)
          pen = true
        }
      }
      ctx.stroke()
    }
    ctx.restore()
  }

  /**
   * Plans the flight for the page as it now stands, outlines in the page's own
   * coordinates the hero's ridge the lines are clipped below and the ground
   * they land on, and sets the pen at the sample the page's scroll points to.
   */
  fit(g: Geometry): void {
    const run    = flight.plan(g, this.#sky.horizon, this.#fire)
    this.#fitted = {
      clip   : groundPath(g.page + 10, x => this.#sky.ridge(x) + 4, g.w),
      ground : groundPath(g.page + 10, run.ridge, g.w),
      run    : run
    }
    this.#shown = this.#target(g, run)
    this.#pace  = 0
  }

  /**
   * Paints the roost, the lines behind the hero's ridge up to the pen, the last
   * stretch darker, the stretches where a bird felt the heat in the fire's
   * color, the fire, and the birds at the pen.
   */
  paint(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number): void {
    const fitted = this.#fitted
    if (fitted === null) return
    const { run } = fitted
    if (g.still) this.#shown = this.#target(g, run)
    const last = Math.max(1, Math.floor(this.#shown))
    this.#roost(ctx, g, fitted)
    if (g.doc !== null) {
      this.#column.x = g.doc.left - 16
      this.#column.y = g.doc.top - g.dy - 12
      this.#column.w = g.doc.right - g.doc.left + 32
      this.#column.h = g.doc.bottom - g.doc.top + 24
    }
    this.#trails(ctx, g, fitted, last)
    const fy = run.fire.y + 30 - g.dy
    if (fy > -120 && fy < g.h + 120) {
      this.#mark.scheme = g.scheme
      this.#mark.x      = run.fire.x
      this.#mark.y      = fy
      this.#marks.fire(ctx, this.#mark)
    }
    this.#heads(ctx, g, dpr, run)
  }

  /**
   * Moves the pen toward the sample the page's scroll points to on a critically
   * damped spring, settling it once it is near enough.
   */
  step(dt: number, g: Geometry): void {
    if (this.#fitted === null) return
    const want   = this.#target(g, this.#fitted.run)
    const omega  = 4.2
    this.#pace  += (omega * omega * (want - this.#shown) - 2 * omega * this.#pace) * dt
    this.#shown += this.#pace * dt
    if (Math.abs(want - this.#shown) < 0.05 && Math.abs(this.#pace) < 0.5) {
      this.#shown = want
      this.#pace  = 0
    }
  }
}
