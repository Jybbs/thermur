import * as chart     from './chart.ts'
import type { Rule } from './measured.ts'

/**
 * Where the heading chart sets each of its parts at one width: the dial's
 * center, outer radius, ring width, and each ring's radius, whether it lays
 * itself out narrow, the column its readings start at, each reading's height,
 * the dial's height and the chart's, and its width.
 */
export interface Layout {
  center : readonly [number, number]
  dial   : number
  height : number
  narrow : boolean
  outer  : number
  radii  : readonly number[]
  readX  : number
  ring   : number
  tall   : number
  width  : number
}
/**
 * What one rule's ring and reading hold once drawn: the ring's arc and the
 * band of its standard error, the angle and the share read beside it with the
 * share's bar and whisker, the groups the rule lifts through, and the bar's
 * full width.
 */
export interface Ring {
  angle   : SVGElement
  arc     : SVGElement
  band    : SVGElement
  bar     : SVGElement
  groups  : readonly SVGElement[]
  share   : SVGElement
  whisker : SVGElement
  width   : number
}
/**
 * How the chart lays itself out and moves: how far behind the ring outside it
 * each ring starts its sweep, as a share of the sweep, the narrowest width it
 * sets the readings beside the dial at, and how many settling times a sweep
 * takes.
 */
export interface Tuning {
  lag    : number
  narrow : number
  sweep  : number
}

/**
 * How the band of a rule's standard error is stroked along its ring.
 */
const BAND = { 'opacity': 0.55, 'stroke-linecap': 'round', 'stroke-width': 3 } as const
/**
 * How the chart lays itself out and moves unless told otherwise.
 */
export const DIAL: Tuning = { lag: 0.14, narrow: 520, sweep: 5 }
/**
 * The name each heading rule reads under, the default's first, beside the
 * shorter name a narrow chart sets.
 */
export const NAMES = {
  reference : ['Overdamped, the default', 'Overdamped'],
  consensus : ['Consensus', 'Consensus'],
  delayed   : ['Delayed', 'Delayed']
} as const
/**
 * How the ring of each stop on the dial's rim is drawn.
 */
const RIM = { 'fill': chart.TONE.ground, 'r': 7, 'stroke': chart.TONE.lead, 'stroke-width': 1.5 } as const
/**
 * The angles the dial marks with a spoke, in degrees.
 */
const SPOKES = [0, 30, 60, 90] as const

/**
 * Draws the rule at `row`'s reading beside the dial into `plot`: its name, the
 * angle and the share its farthest drones turn through, and the share's bar
 * over its track with the whisker of its standard error.
 */
function drawReading(
  layout : Layout,
  plot   : SVGElement,
  row    : number,
  rule   : Rule
): Omit<Ring, 'arc' | 'band' | 'groups'> & { group: SVGElement } {
  const { center, dial, narrow, outer, readX, tall, width } = layout
  const top     = narrow ? dial + row * tall : center[1] - outer + 4 + row * tall
  const barX    = narrow ? readX + 100 : readX
  const barY    = narrow ? top + 8 : top + 46
  const length  = narrow ? Math.min(150, width - barX - 16) : Math.min(220, width - readX - 120)
  const angle   = chart.text(chart.FACE.mono, 21, '', readX, top + 38, { 'font-weight': 500 })
  const share   = chart.text(chart.FACE.mono, 11.5, '', readX + 76, top + 37, { fill: chart.TONE.muted })
  const bar     = chart.svg('rect', { fill: rule.color, height: 4, rx: 2, width: 0, x: barX, y: barY })
  const tick    = { 'opacity': 0.5, 'stroke': chart.TONE.ink, 'stroke-width': 1 }
  const whisker = chart.svg('line', { ...tick, y1: barY + 2, y2: barY + 2 })
  const named   = { 'fill': rule.color, 'font-weight': row === 0 ? 600 : 400 }
  const group   = chart.svg('g', { class: 'rules-fade' },
    chart.svg('rect', { fill: 'transparent', height: tall - 4, width: width - readX, x: readX - 6, y: top }),
    chart.text(chart.FACE.body, 14, narrow ? rule.short : rule.label, readX, top + 14, named),
    angle,
    share,
    chart.svg('rect', { fill: rule.color, height: 4, opacity: 0.15, rx: 2, width: length, x: barX, y: barY }),
    bar,
    whisker)
  plot.append(group)
  return { angle, bar, group, share, whisker, width: length }
}

/**
 * Writes the path of an arc `radius` from the dial's center, from `from`
 * degrees to `to`, a sliver where the two meet.
 */
export function arc(layout: Layout, radius: number, from: number, to: number): string {
  const [x0, y0] = at(layout, radius, from)
  const [x1, y1] = at(layout, radius, Math.max(to, from + 0.01))
  return `M${x0},${y0} A${radius},${radius} 0 0 0 ${x1},${y1}`
}

/**
 * Finds the point `radius` from the dial's center at `degrees` above the
 * horizontal.
 */
export function at(layout: Layout, radius: number, degrees: number): [number, number] {
  const angle = degrees * Math.PI / 180
  return [layout.center[0] + radius * Math.cos(angle), layout.center[1] - radius * Math.sin(angle)]
}

/**
 * Draws the dial's frame into `plot`, a spoke at each marked angle under its
 * rim and its caption, and returns the dashed marker of the first drone's
 * turn.
 */
export function drawFrame(layout: Layout, plot: SVGElement): SVGElement {
  const { outer, radii, ring } = layout
  for (const degrees of SPOKES) {
    const [x1, y1] = at(layout, radii.at(-1)! - ring / 2 - 4, degrees)
    const [x2, y2] = at(layout, outer + 2, degrees)
    plot.append(chart.svg('line', { 'stroke': chart.TONE.rule, 'stroke-width': 1, x1, x2, y1, y2 }))
  }
  const rim   = { 'fill': 'none', 'stroke': chart.TONE.rule, 'stroke-width': 1 }
  const named = { 'fill': chart.TONE.muted, 'letter-spacing': '0.08em' }
  plot.append(
    chart.svg('path', { ...rim, d: arc(layout, outer + 6, 0, 90) }),
    chart.text(chart.FACE.mono, 10.5, 'THE FIRST DRONE TURNS', 48, 14, named)
  )
  return chart.svg('path', {
    'fill'             : 'none',
    'stroke'           : chart.TONE.ink,
    'stroke-dasharray' : '5 4',
    'stroke-width'     : 1.5
  })
}

/**
 * Draws the stop for the turn of `turn` degrees on the dial's rim, a radio
 * whose dot shows while it is picked.
 */
export function drawKnob(layout: Layout, turn: number): SVGElement {
  const [x, y]   = at(layout, layout.outer + 6, turn)
  const [lx, ly] = at(layout, layout.outer + 24, turn)
  return chart.svg('g', { 'aria-label': `${turn}°`, 'class': 'rules-knob', 'role': 'radio' },
    chart.svg('circle', { ...RIM, cx: x, cy: y }),
    chart.svg('circle', { class: 'rules-knob-dot', cx: x, cy: y, fill: chart.TONE.lead, r: 3.5 }),
    chart.text(chart.FACE.mono, 11.5, `${turn}°`, lx, ly + 4, { 'text-anchor': 'middle' }))
}

/**
 * Draws the rule at `row` into `plot`: its ring on the dial over a faint
 * quarter, the band of its standard error along the ring's edge, and its
 * reading beside the dial.
 */
export function drawRing(layout: Layout, plot: SVGElement, row: number, rule: Rule): Ring {
  const stroke              = { 'fill': 'none', 'stroke': rule.color, 'stroke-width': layout.ring }
  const path                = chart.svg('path', stroke)
  const band                = chart.svg('path', { ...stroke, ...BAND })
  const quarter             = arc(layout, layout.radii[row]!, 0, 90)
  const backing             = chart.svg('path', { ...stroke, d: quarter, opacity: 0.1 })
  const arcs                = chart.svg('g', { class: 'rules-fade' }, backing, path, band)
  plot.append(arcs)
  const { group, ...shown } = drawReading(layout, plot, row, rule)
  return { ...shown, arc: path, band, groups: [arcs, group] }
}

/**
 * Lays the chart out at `width` for `count` rules.
 */
export function layoutOf(count: number, tuning: Tuning, width: number): Layout {
  const narrow = width < tuning.narrow
  const outer  = narrow ? Math.min(width - 64, 250) : Math.min(230, Math.round(width * 0.36))
  const ring   = Math.min(24, outer / 7)
  const tall   = narrow ? 52 : Math.max(58, (outer - 10) / count)
  const dial   = outer + 44
  return {
    center : [22, outer + 34],
    dial,
    height : narrow ? dial + count * tall + 4 : dial,
    narrow,
    outer,
    radii  : Array.from({ length: count }, (_, row) => outer - row * (ring + 8) - ring / 2),
    readX  : narrow ? 8 : 22 + outer + 64,
    ring,
    tall,
    width
  }
}
