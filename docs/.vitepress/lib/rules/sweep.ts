import * as chart    from './chart.ts'
import * as measured from './measured.ts'
import * as motion   from './motion.ts'
import * as spacing  from './spacing.ts'

/**
 * What each key does to the track's handle, a step between the measured
 * spacings or a move to either end.
 */
const KEYS: Readonly<Record<string, (at: number, last: number) => number>> = {
  ArrowDown  : at => at - 1,
  ArrowLeft  : at => at - 1,
  ArrowRight : at => at + 1,
  ArrowUp    : at => at + 1,
  End        : (_, last) => last,
  Home       : () => 0
}

/**
 * Does nothing, standing in for a transition before the first one starts.
 */
const idle = (): void => {}

/**
 * The coupling chart: each rule's polarization on a row of its own against the
 * launch spacing set on a track drawn into the chart. Dragging the track's
 * handle glides every rule's dot along the straight line between the measured
 * spacings and a release settles it on the nearest one, the arrow keys and
 * Home and End step it between them, and Play sweeps it from the densest
 * launch to the sparsest and back. Each rule's value is read at the end of its
 * row, a dashed tie runs from each dot to the default's value with the widest
 * gap labeled, and a pointer over a row lifts it. `destroy` releases every
 * listener, observer, and frame it holds.
 */
export class Sweep {
  #frame             = 0
  #gap               : SVGElement | undefined
  #guide             : SVGElement | undefined
  readonly #host     : HTMLElement
  #last              = 0
  #layout            : spacing.Layout | undefined
  #lifted            = -1
  readonly #moving   : () => void
  #phase             = 0
  #rail              : spacing.Rail | undefined
  readonly #released : () => void
  #rows              : spacing.Row[] = []
  readonly #rules    : readonly measured.Rule[]
  #spacing           : number
  #stop              = idle
  readonly #stops    : readonly number[]
  readonly #tuning   : spacing.Tuning

  constructor(host: HTMLElement, rules: readonly measured.Rule[], tuning: spacing.Tuning = spacing.TRACK) {
    this.#host     = host
    this.#rules    = rules
    this.#tuning   = tuning
    this.#stops    = rules[0]!.points.map(point => point.given)
    this.#spacing  = this.#stops[0]!
    this.#released = motion.widthOf(host, width => this.#draw(width))
    this.#moving   = motion.onMotion(() => {
      if (this.#frame && motion.still()) this.#settle()
    })
  }

  /**
   * Lays the chart out at `width` and draws it whole, the track, the axis, and
   * a row per rule, then wires the track.
   */
  #draw(width: number): void {
    const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
    const layout   = spacing.layoutOf(this.#rules.length, hi, lo, this.#tuning, width)
    const plot     = chart.svg('svg', {
      'aria-label' : 'Polarization of each coupling rule against the launch spacing set on the track',
      'class'      : 'rules-chart',
      'height'     : layout.height,
      'role'       : 'group',
      'viewBox'    : `0 0 ${width} ${layout.height}`,
      'width'      : width
    })
    this.#layout   = layout
    this.#rail     = spacing.drawRail(layout, this.#frame !== 0, plot, this.#spacing, this.#stops)
    this.#guide    = spacing.drawAxis(layout, plot)
    this.#rows     = this.#rules.map((rule, place) => spacing.drawRow(layout, place, plot, rule))
    this.#gap      = chart.text(chart.FACE.mono, 10.5, '', 0, 0, { ...spacing.MUTED, ...spacing.MIDDLE })
    plot.append(this.#gap)
    this.#rows.forEach((row, place) => {
      row.group.addEventListener('pointerenter', () => this.#lift(place))
      row.group.addEventListener('pointerleave', () => this.#lift(-1))
    })
    this.#wire(plot, this.#rail)
    this.#host.replaceChildren(plot)
    this.#show(this.#spacing)
    this.#lift(this.#lifted)
  }

  /**
   * Glides the track's handle from where it stands to `to`.
   */
  #glide(to: number): void {
    const from = this.#spacing
    this.#stop()
    this.#stop = motion.tween(share => this.#show(from + (to - from) * share), 2)
  }

  /**
   * Stops the sweep, setting the knob back to play.
   */
  #halt(): void {
    cancelAnimationFrame(this.#frame)
    this.#frame = 0
    this.#rail?.player.setAttribute('aria-pressed', 'false')
    this.#rail?.glyph.setAttribute('d', spacing.GLYPH.play)
  }

  /**
   * Lifts the row at `row` over the others, or none where `row` is -1.
   */
  #lift(row: number): void {
    this.#lifted = row
    this.#rows.forEach((each, place) =>
      chart.set(each.group, { opacity: row < 0 || place === row ? 1 : 0.35 }))
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
  #play(): void {
    if (this.#frame) {
      this.#settle()
      return
    }
    if (motion.still()) return
    const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
    this.#stop()
    this.#phase    = Math.acos(1 - 2 * (this.#spacing - lo) / (hi - lo)) / (Math.PI * 2)
    this.#last     = 0
    this.#frame    = requestAnimationFrame(this.#run)
    this.#rail?.player.setAttribute('aria-pressed', 'true')
    this.#rail?.glyph.setAttribute('d', spacing.GLYPH.pause)
  }

  /**
   * Moves the sweep on to the frame at `now`, settling it where motion has
   * come to stand still.
   */
  readonly #run = (now: number): void => {
    if (motion.still()) {
      this.#settle()
      return
    }
    const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
    const dt       = this.#last ? Math.min(0.1, (now - this.#last) / 1000) : 0
    this.#last     = now
    this.#phase    = (this.#phase + dt / this.#tuning.sweep) % 1
    this.#show(lo + (hi - lo) * (0.5 - 0.5 * Math.cos(this.#phase * Math.PI * 2)))
    this.#frame    = requestAnimationFrame(this.#run)
  }

  /**
   * Stops the sweep and glides the handle to the nearest measured spacing.
   */
  #settle(): void {
    this.#halt()
    this.#glide(this.#nearest(this.#spacing))
  }

  /**
   * Sets the chart to `value`: each rule's dot, tie, and reading, the handle
   * and the spacing above it, the default's guide, and the widest gap from it.
   */
  #show(value: number): void {
    const { fromS, toX } = this.#layout!
    const rail           = this.#rail!
    const stopped        = this.#stops.some(each => Math.abs(each - value) < 0.05)
    const points         = this.#rules.map(rule => measured.between(rule, value))
    const base           = points[0]!.mean
    const widest         = points.reduce((best, point, row) =>
      row > 0 && point.mean < points[best]!.mean ? row : best, 1)
    const words          = `${value.toFixed(stopped ? 0 : 1)} m`
    this.#spacing        = value
    this.#rows.forEach((row, place) => {
      const point = points[place]!
      const x     = toX(point.mean)
      chart.set(row.dot, { cx: x })
      chart.set(row.tie, { x1: Math.min(x, toX(base)), x2: Math.max(x, toX(base)) })
      chart.set(row.value, { opacity: stopped ? 1 : 0.6 })
      row.value.textContent = stopped ? measured.reading(point) : `≈ ${point.mean.toFixed(2)}`
    })
    chart.set(rail.handle, {
      'aria-valuenow'  : value.toFixed(1),
      'aria-valuetext' : stopped ? words : `${words}, between measured spacings`,
      'cx'             : fromS(value)
    })
    chart.set(rail.fill, { x2: fromS(value) })
    chart.set(rail.readout, { x: fromS(value) })
    chart.set(this.#guide!, { x1: toX(base), x2: toX(base) })
    chart.set(this.#gap!, {
      x : (toX(base) + toX(points[widest]!.mean)) / 2,
      y : Number(this.#rows[widest]!.tie.getAttribute('y1')) - 7
    })
    rail.readout.textContent = words
    this.#gap!.textContent   = `−${(base - points[widest]!.mean).toFixed(2)}`
  }

  /**
   * Wires the track: a press on the strip or the handle holds it and a drag
   * moves it, a release settles it, the keys step it, and the knob plays the
   * sweep.
   */
  #wire(plot: SVGElement, rail: spacing.Rail): void {
    const { hit, handle, player } = rail
    const drag                    = (event: PointerEvent): void => {
      const [lo, hi] = [this.#stops[0]!, this.#stops.at(-1)!]
      const x        = event.clientX - plot.getBoundingClientRect().left
      this.#show(Math.min(hi, Math.max(lo, this.#layout!.toS(x))))
    }
    const hold                    = (event: PointerEvent): void => {
      this.#halt()
      this.#stop()
      hit.setPointerCapture(event.pointerId)
    }
    hit.addEventListener('pointerdown', event => {
      hold(event)
      drag(event)
    })
    hit.addEventListener('pointermove', event => {
      if (hit.hasPointerCapture(event.pointerId)) drag(event)
    })
    hit.addEventListener('pointerup', () => this.#glide(this.#nearest(this.#spacing)))
    handle.addEventListener('pointerdown', hold)
    handle.addEventListener('keydown', event => {
      const move = KEYS[event.key]
      if (move === undefined) return
      const last = this.#stops.length - 1
      const at   = move(this.#stops.indexOf(this.#nearest(this.#spacing)), last)
      event.preventDefault()
      this.#halt()
      this.#glide(this.#stops[Math.max(0, Math.min(last, at))]!)
    })
    player.addEventListener('click', () => this.#play())
    player.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return
      event.preventDefault()
      this.#play()
    })
  }

  destroy(): void {
    this.#halt()
    this.#stop()
    this.#released()
    this.#moving()
  }
}
