import * as chart         from './chart.ts'
import * as dial          from './dial.ts'
import type { Rule }      from './measured.ts'
import { tween, widthOf } from './motion.ts'

/**
 * What each key does to the turn picked on the dial's rim, a step to the next
 * turn or a press of the one in focus.
 */
const KEYS: Readonly<Record<string, -1 | 0 | 1>> = {
  ' '          : 0,
  'ArrowDown'  : -1,
  'ArrowLeft'  : -1,
  'ArrowRight' : 1,
  'ArrowUp'    : 1,
  'Enter'      : 0
}

/**
 * Does nothing, standing in for a sweep before the first one starts.
 */
const idle = (): void => {}

/**
 * The heading chart: a quarter dial with a ring per rule, each sweeping
 * through the angle the farthest fifth of the flock turns while the first
 * drone turns through the angle picked on the stops at the dial's rim, each
 * ring a moment after the one outside it, with its standard error drawn as a
 * band along its edge. Each rule's angle, its share of the turn, and that
 * share's bar are read beside the dial, and a pointer over a ring or its
 * reading lifts that rule. `destroy` releases every listener, observer, and
 * frame it holds.
 */
export class Reach {
  #errors            : number[]
  readonly #host     : HTMLElement
  #index             = 0
  #knobs             : SVGElement[] = []
  #layout            : dial.Layout | undefined
  #marker            : SVGElement | undefined
  readonly #released : () => void
  #rings             : dial.Ring[] = []
  readonly #rules    : readonly Rule[]
  #shown             : number[]
  #stop              = idle
  #turn              = 0
  readonly #turns    : readonly number[]
  readonly #tuning   : dial.Tuning

  constructor(host: HTMLElement, rules: readonly Rule[], tuning: dial.Tuning = dial.DIAL) {
    this.#host     = host
    this.#rules    = rules
    this.#tuning   = tuning
    this.#turns    = rules[0]!.points.map(point => point.given)
    this.#shown    = rules.map(() => 0)
    this.#errors   = rules.map(() => 0)
    this.#released = widthOf(host, width => this.#draw(width))
    this.#go(0)
  }

  /**
   * Marks the turn at `next` picked on the rim, moving the focus to it where
   * `focus` is set.
   */
  #choose(next: number, focus = false): void {
    this.#knobs.forEach((knob, place) => {
      knob.setAttribute('aria-checked', String(place === next))
      knob.setAttribute('tabindex', place === next ? '0' : '-1')
      if (place === next && focus) knob.focus()
    })
  }

  /**
   * Lays the chart out at `width` and draws it whole, the frame, a ring and a
   * reading per rule, and a stop on the rim per turn, then wires the stops.
   */
  #draw(width: number): void {
    const layout = dial.layoutOf(this.#rules.length, this.#tuning, width)
    const plot   = chart.svg('svg', {
      'aria-label' : 'The angle the farthest drones turn through, against the turn the first drone makes, '
                   + 'for each heading rule',
      'class'      : 'rules-chart',
      'height'     : layout.height,
      'role'       : 'group',
      'viewBox'    : `0 0 ${width} ${layout.height}`,
      'width'      : width
    })
    const group  = chart.svg('g', { 'aria-label': 'The first drone turns', 'role': 'radiogroup' })
    this.#layout = layout
    this.#marker = dial.drawFrame(layout, plot)
    this.#rings  = this.#rules.map((rule, row) => dial.drawRing(layout, plot, row, rule))
    this.#knobs  = this.#turns.map(turn => dial.drawKnob(layout, turn))
    this.#rings.forEach(({ groups }, row) => {
      for (const each of groups) {
        each.addEventListener('pointerenter', () => this.#lift(row))
        each.addEventListener('pointerleave', () => this.#lift(-1))
      }
    })
    this.#knobs.forEach((knob, place) => {
      knob.addEventListener('click', () => this.#go(place))
      knob.addEventListener('keydown', event => {
        const step = KEYS[event.key]
        if (step === undefined) return
        const next = (place + step + this.#turns.length) % this.#turns.length
        event.preventDefault()
        this.#go(next)
        this.#choose(next, true)
      })
    })
    group.append(...this.#knobs)
    plot.append(this.#marker, group)
    this.#host.replaceChildren(plot)
    this.#choose(this.#index)
    this.#place()
  }

  /**
   * Picks the turn at `next`, sweeping each ring, its band, and the marker of
   * the first drone's turn from where they stand to the new turn's values.
   */
  #go(next: number): void {
    const { lag, sweep } = this.#tuning
    const from           = [...this.#shown]
    const spread         = [...this.#errors]
    const turned         = this.#turn
    const to             = this.#rules.map(rule => rule.points[next]!.mean * rule.points[next]!.given)
    const errors         = this.#rules.map(rule => rule.points[next]!.error * rule.points[next]!.given)
    const room           = 1 - lag * (this.#rules.length - 1)
    this.#index          = next
    this.#choose(next)
    this.#stop()
    this.#stop           = tween(share => {
      this.#shown  = from.map((value, row) =>
        value + (to[row]! - value) * Math.min(1, Math.max(0, (share - lag * row) / room)))
      this.#errors = spread.map((value, row) => value + (errors[row]! - value) * share)
      this.#turn   = turned + (this.#turns[next]! - turned) * Math.min(1, share / room)
      this.#place()
    }, sweep)
  }

  /**
   * Lifts the rule at `row` over the others, or none where `row` is -1.
   */
  #lift(row: number): void {
    this.#rings.forEach(({ groups }, place) => {
      for (const each of groups) chart.set(each, { opacity: row < 0 || place === row ? 1 : 0.3 })
    })
  }

  /**
   * Sets each ring, band, and reading to the values the sweep stands at, and
   * the marker to the first drone's turn.
   */
  #place(): void {
    const layout = this.#layout
    if (layout === undefined) return
    const turned = Math.max(this.#turn, 1e-6)
    this.#rings.forEach((ring, row) => {
      const shown  = this.#shown[row]!
      const spread = this.#errors[row]!
      const share  = shown / turned
      const x0     = Number(ring.bar.getAttribute('x'))
      const error  = this.#rules[row]!.points[this.#index]!.error
      const radius = layout.radii[row]!
      const edge   = radius + layout.ring / 2 + 3
      chart.set(ring.arc, { d: dial.arc(layout, radius, 0, shown) })
      chart.set(ring.band, { d: dial.arc(layout, edge, Math.max(0, shown - spread), shown + spread) })
      chart.set(ring.bar, { width: Math.max(0, ring.width * Math.min(1, share)) })
      chart.set(ring.whisker, {
        x1 : x0 + ring.width * Math.max(0, share - spread / turned),
        x2 : x0 + ring.width * Math.min(1.1, share + spread / turned)
      })
      ring.angle.textContent = `${shown.toFixed(1)}°`
      ring.share.textContent = `of ${Math.round(this.#turn)}° · `
                             + `${Math.round(share * 100)}% ± ${Math.round(error * 100)}`
    })
    const [x, y] = dial.at(layout, layout.outer + 6, this.#turn)
    chart.set(this.#marker!, { d: `M${layout.center[0]},${layout.center[1]} L${x},${y}` })
  }

  destroy(): void {
    this.#stop()
    this.#released()
  }
}
