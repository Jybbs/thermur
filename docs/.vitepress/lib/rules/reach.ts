import { idle, markup, tween } from '@thermur/kit'

import * as common   from '../common.ts'
import * as dial     from './dial.ts'
import type { Rule } from './measured.ts'

/**
 * The heading chart: a quarter dial with a ring per rule, each sweeping
 * through the angle the farthest fifth of the flock turns while the first
 * drone turns through the angle `go` picks, each ring a moment after the one
 * outside it, with its standard error drawn as a band along its edge. A stop
 * per turn is drawn on the dial's rim, its dot showing while its turn is
 * picked, and `placed` hears where each stop stands at each width, so the
 * radios that pick a turn can stand over them. Each rule's angle, its share of
 * the turn, and that share's bar are read beside the dial, and a pointer over
 * a ring or its reading lifts that rule. `destroy` releases every listener,
 * observer, and frame it holds.
 */
export class Reach {
  #errors            : number[]
  readonly #host     : HTMLElement
  #index             = 0
  #knobs             : SVGElement[] = []
  #layout            : dial.Layout | undefined
  #marker            : SVGElement | undefined
  readonly #placed   : (stops: readonly markup.Spot[]) => void
  readonly #released : () => void
  #rings             : dial.Ring[] = []
  readonly #rules    : readonly Rule[]
  #shown             : number[]
  #stop              = idle
  #turn              = 0
  readonly #turns    : readonly number[]
  readonly #tuning   : dial.Tuning

  constructor(
    host   : HTMLElement,
    placed : (stops: readonly markup.Spot[]) => void,
    rules  : readonly Rule[],
    tuning : dial.Tuning = dial.DIAL
  ) {
    this.#host     = host
    this.#placed   = placed
    this.#rules    = rules
    this.#tuning   = tuning
    this.#turns    = rules[0]!.points.map(point => point.given)
    this.#shown    = rules.map(() => 0)
    this.#errors   = rules.map(() => 0)
    this.#released = markup.widthOf(host, width => this.#draw(width))
    this.go(0)
  }

  /**
   * Shows the dot of the stop at `next` on the rim, and only that one.
   */
  #choose(next: number): void {
    this.#knobs.forEach((knob, place) => markup.set(knob, { 'data-on': String(place === next) }))
  }

  /**
   * Lays the chart out at `width` and draws it whole, the frame, a ring and a
   * reading per rule, and a stop on the rim per turn, then tells `placed`
   * where the stops stand.
   */
  #draw(width: number): void {
    const layout = dial.layoutOf(this.#rules.length, this.#tuning, width)
    const plot   = markup.svg('svg', {
      'aria-label' : 'The angle the farthest drones turn through, against the turn the first drone makes, '
                   + 'for each heading rule',
      'class'      : 'rules-chart',
      'height'     : layout.height,
      'role'       : 'group',
      'viewBox'    : `0 0 ${width} ${layout.height}`,
      'width'      : width
    })
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
    plot.append(this.#marker, ...this.#knobs)
    this.#host.replaceChildren(plot)
    this.#choose(this.#index)
    this.#place()
    this.#placed(this.#turns.map(turn => {
      const [left, top] = dial.at(layout, layout.outer + 6, turn)
      return { left, top }
    }))
  }

  /**
   * Picks the turn at `next`, sweeping each ring, its band, and the marker of
   * the first drone's turn from where they stand to the new turn's values.
   */
  go(next: number): void {
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
        common.lerp(value, to[row]!, common.clamp((share - lag * row) / room)))
      this.#errors = spread.map((value, row) => common.lerp(value, errors[row]!, share))
      this.#turn   = common.lerp(turned, this.#turns[next]!, common.clamp(share / room))
      this.#place()
    }, sweep)
  }

  /**
   * Lifts the rule at `row` over the others, or none where `row` is -1.
   */
  #lift(row: number): void {
    this.#rings.forEach(({ groups }, place) => {
      for (const each of groups) markup.set(each, { opacity: row < 0 || place === row ? 1 : 0.3 })
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
      markup.set(ring.arc, { d: dial.arc(layout, radius, 0, shown) })
      markup.set(ring.band, { d: dial.arc(layout, edge, Math.max(0, shown - spread), shown + spread) })
      markup.set(ring.bar, { width: Math.max(0, ring.width * Math.min(1, share)) })
      markup.set(ring.whisker, {
        x1 : x0 + ring.width * Math.max(0, share - spread / turned),
        x2 : x0 + ring.width * Math.min(1.1, share + spread / turned)
      })
      ring.angle.textContent = `${shown.toFixed(1)}°`
      ring.share.textContent = `of ${Math.round(this.#turn)}° · `
                             + `${Math.round(share * 100)}% ± ${Math.round(error * 100)}`
    })
    const [x, y] = dial.at(layout, layout.outer + 6, this.#turn)
    markup.set(this.#marker!, { d: `M${layout.center[0]},${layout.center[1]} L${x},${y}` })
  }

  destroy(): void {
    this.#stop()
    this.#released()
  }
}
