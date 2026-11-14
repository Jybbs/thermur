import { Frames, idle, markup, tween } from '@thermur/kit'

import * as common   from '../common.ts'
import * as measured from './measured.ts'
import * as spacing  from './spacing.ts'

/**
 * What a sweep tells the controls standing over its track: where the knob that
 * plays it and the groove its handle runs along stand at each width, each
 * spacing it shows, and whether it plays.
 */
export interface View {
  moved  : (value: number) => void
  placed : (rail: Pick<spacing.Rail, 'player' | 'track'>) => void
  played : (on: boolean) => void
}

/**
 * Finds whether `value` stands on one of the measured spacings `stops`.
 */
const onStop = (stops: readonly number[], value: number): boolean =>
  stops.some(each => Math.abs(each - value) < 0.05)

/**
 * Words the spacing `value` for a screen reader, beside whether it stands
 * between two of the measured spacings `stops`.
 */
export function spoken(stops: readonly number[], value: number): string {
  const stopped = onStop(stops, value)
  const words   = `${value.toFixed(stopped ? 0 : 1)} m`
  return stopped ? words : `${words}, between measured spacings`
}

/**
 * The coupling chart: each rule's polarization on a row of its own against the
 * launch spacing set on a track drawn into the chart. `drag` sets every rule's
 * dot at once on the straight line between the measured spacings, `settle`
 * glides it to the measured spacing nearest a value, and `play` sweeps it from
 * the densest launch to the sparsest and back, each reported through `view`, so
 * the controls standing over the track follow it. Each rule's value is read at
 * the end of its row, a dashed tie runs from each dot to the default's value
 * with the widest gap labeled, and a pointer over a row lifts it. `destroy`
 * releases every listener, observer, and frame it holds.
 */
export class Sweep {
  #gap               : SVGElement | undefined
  #guide             : SVGElement | undefined
  readonly #host     : HTMLElement
  #layout            : spacing.Layout | undefined
  #lifted            = -1
  readonly #loop     : Frames
  #phase             = 0
  #rail              : spacing.Rail | undefined
  readonly #released : () => void
  #rows              : spacing.Row[] = []
  readonly #rules    : readonly measured.Rule[]
  #spacing           : number
  #stop              = idle
  readonly #stops    : readonly number[]
  readonly #tuning   : spacing.Tuning
  readonly #view     : View

  constructor(
    host   : HTMLElement,
    rules  : readonly measured.Rule[],
    view   : View,
    tuning : spacing.Tuning = spacing.TRACK
  ) {
    this.#host     = host
    this.#view     = view
    this.#rules    = rules
    this.#tuning   = tuning
    this.#stops    = rules[0]!.points.map(point => point.given)
    this.#spacing  = this.#stops[0]!
    this.#released = markup.widthOf(host, width => this.#draw(width))
    this.#loop     = new Frames(seconds => this.#advance(seconds), (standing) => {
      if (standing && this.#loop.running) this.settle(this.#spacing)
    })
  }

  /**
   * Lays the chart out at `width` and draws it whole, the track, the axis, and
   * a row per rule, then tells the view where the track stands.
   */
  #draw(width: number): void {
    const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
    const layout   = spacing.layoutOf(this.#rules.length, hi, lo, this.#tuning, width)
    const plot     = markup.svg('svg', {
      'aria-label' : 'Polarization of each coupling rule against the launch spacing set on the track',
      'class'      : 'rules-chart',
      'height'     : layout.height,
      'role'       : 'group',
      'viewBox'    : `0 0 ${width} ${layout.height}`,
      'width'      : width
    })
    this.#layout   = layout
    this.#rail     = spacing.drawRail(layout, this.#loop.running, plot, this.#spacing, this.#stops)
    this.#guide    = spacing.drawAxis(layout, plot)
    this.#rows     = this.#rules.map((rule, place) => spacing.drawRow(layout, place, plot, rule))
    this.#gap      = markup.text(markup.FACE.mono, 10.5, '', 0, 0, { ...spacing.MUTED, ...spacing.MIDDLE })
    plot.append(this.#gap)
    this.#rows.forEach((row, place) => {
      row.group.addEventListener('pointerenter', () => this.#lift(place))
      row.group.addEventListener('pointerleave', () => this.#lift(-1))
    })
    this.#host.replaceChildren(plot)
    this.#show(this.#spacing)
    this.#lift(this.#lifted)
    this.#view.placed(this.#rail)
  }

  /**
   * Glides the track's handle from where it stands to `to`.
   */
  #glide(to: number): void {
    const from = this.#spacing
    this.#stop()
    this.#stop = tween(share => this.#show(common.lerp(from, to, share)), 2)
  }

  /**
   * Stops the sweep, setting the knob back to play.
   */
  #halt(): void {
    this.#loop.run(false)
    this.#rail?.glyph.setAttribute('d', spacing.GLYPH.play)
    this.#view.played(false)
  }

  /**
   * Lifts the row at `row` over the others, or none where `row` is -1.
   */
  #lift(row: number): void {
    this.#lifted = row
    this.#rows.forEach((each, place) =>
      markup.set(each.group, { opacity: row < 0 || place === row ? 1 : 0.35 }))
  }

  /**
   * Finds the measured spacing nearest `value`.
   */
  #nearest(value: number): number {
    return this.#stops.reduce((best, each) => Math.abs(each - value) < Math.abs(best - value) ? each : best)
  }

  /**
   * Starts the sweep from where the handle stands, or stops it and settles
   * the handle on the nearest measured spacing where it runs, starting
   * nothing for a reader whose motion stands still.
   */
  play(): void {
    if (this.#loop.running) {
      this.settle(this.#spacing)
      return
    }
    const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
    this.#phase    = Math.acos(1 - 2 * (this.#spacing - lo) / (hi - lo)) / (Math.PI * 2)
    if (!this.#loop.run(true)) return
    this.#stop()
    this.#rail?.glyph.setAttribute('d', spacing.GLYPH.pause)
    this.#view.played(true)
  }

  /**
   * Moves the sweep on by `seconds`, at most a tenth of a second, so a frame
   * that comes late skips no part of the sweep.
   */
  #advance(seconds: number): void {
    const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
    this.#phase    = (this.#phase + Math.min(0.1, seconds) / this.#tuning.sweep) % 1
    this.#show(lo + (hi - lo) * (0.5 - 0.5 * Math.cos(this.#phase * Math.PI * 2)))
  }

  /**
   * Stops the sweep and glides the handle to the measured spacing nearest
   * `value`.
   */
  settle(value: number): void {
    this.#halt()
    this.#glide(this.#nearest(value))
  }

  /**
   * Stops the sweep and any glide and sets the chart to `value` at once, as a
   * drag along the track does.
   */
  drag(value: number): void {
    this.#halt()
    this.#stop()
    this.#show(value)
  }

  /**
   * Sets the chart to `value`: each rule's dot, tie, and reading, the handle
   * and the spacing above it, the default's guide, and the widest gap from it.
   */
  #show(value: number): void {
    const { fromS, toX } = this.#layout!
    const rail           = this.#rail!
    const stopped        = onStop(this.#stops, value)
    const points         = this.#rules.map(rule => measured.between(rule, value))
    const base           = points[0]!.mean
    const widest         = points.reduce((best, point, row) =>
      row > 0 && point.mean < points[best]!.mean ? row : best, 1)
    const words          = `${value.toFixed(stopped ? 0 : 1)} m`
    this.#spacing        = value
    this.#rows.forEach((row, place) => {
      const point = points[place]!
      const x     = toX(point.mean)
      markup.set(row.dot, { cx: x })
      markup.set(row.tie, { x1: Math.min(x, toX(base)), x2: Math.max(x, toX(base)) })
      markup.set(row.value, { opacity: stopped ? 1 : 0.6 })
      row.value.textContent = stopped ? measured.reading(point) : `≈ ${point.mean.toFixed(2)}`
    })
    markup.set(rail.handle, { cx: fromS(value) })
    markup.set(rail.fill, { x2: fromS(value) })
    markup.set(rail.readout, { x: fromS(value) })
    markup.set(this.#guide!, { x1: toX(base), x2: toX(base) })
    markup.set(this.#gap!, {
      x : (toX(base) + toX(points[widest]!.mean)) / 2,
      y : Number(this.#rows[widest]!.tie.getAttribute('y1')) - 7
    })
    rail.readout.textContent = words
    this.#gap!.textContent   = `−${(base - points[widest]!.mean).toFixed(2)}`
    this.#view.moved(value)
  }

  destroy(): void {
    this.#halt()
    this.#stop()
    this.#released()
    this.#loop.destroy()
  }
}
