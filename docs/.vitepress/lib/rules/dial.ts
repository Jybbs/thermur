import { markup } from '@thermur/kit'

import type { Rule } from './measured.ts'

/**
 * Where the heading chart sets each of its parts at one width: the dial's
 * center, outer radius, ring width, and each ring's radius, the widest angle
 * its frame reaches, whether it lays itself out narrow or wide, the column its
 * readings start and end at, each reading's height, the chart's height, and
 * its width.
 */
export interface Layout {
  center : readonly [number, number]
  height : number
  narrow : boolean
  outer  : number
  radii  : readonly number[]
  reach  : number
  readTo : number
  readX  : number
  ring   : number
  tall   : number
  wide   : boolean
  width  : number
}
/**
 * What one rule's ring and reading hold once drawn: the ring's arc and the
 * band of its standard error up to the first drone's turn, the lighter arc
 * and band past it, the angle and the share read beside it with the share's
 * bar and whisker, the groups the rule lifts through, and the bar's full
 * width.
 */
export interface Ring {
  angle    : SVGElement
  arc      : SVGElement
  band     : SVGElement
  bar      : SVGElement
  groups   : readonly SVGElement[]
  pastArc  : SVGElement
  pastBand : SVGElement
  share    : SVGElement
  whisker  : SVGElement
  width    : number
}
/**
 * How the chart lays itself out and moves: how far behind the ring outside it
 * each ring starts its sweep, as a share of the sweep, the narrowest width it
 * sets the readings beside the dial at, how many settling times a sweep
 * takes, and the narrowest width it sets each name in full on its own line.
 */
export interface Tuning {
  lag    : number
  narrow : number
  sweep  : number
  wide   : number
}

/**
 * How the band of a rule's standard error is stroked along its ring.
 */
const BAND = { 'opacity': 0.55, 'stroke-linecap': 'round', 'stroke-width': 3 } as const
/**
 * How the chart lays itself out and moves unless told otherwise.
 */
export const DIAL: Tuning = { lag: 0.14, narrow: 600, sweep: 5, wide: 960 }
/**
 * The radius of the disc in the page's ground color drawn under each stop on
 * the rim, clearing a gap between the stop's ring and a band running past it.
 */
const HALO = 10.5
/**
 * The name each heading rule the chart draws reads under, the default's first
 * and the overdamped limit second, beside the shorter name a narrow chart sets.
 */
export const NAMES = {
  reference  : ['Consensus, the default', 'Consensus'],
  overdamped : ['Overdamped limit', 'Overdamped'],
  inertial   : ['Inertial spin', 'Inertial spin'],
  delayed    : ['Delayed', 'Delayed'],
  nonmutual  : ['Nonmutual', 'Nonmutual'],
  quartic    : ['Quartic', 'Quartic']
} as const
/**
 * The heading rule drawn in the ink, the palette holding no hue for it.
 */
export const INKED = 'overdamped'
/**
 * The opacity a ring and its band take past the first drone's turn.
 */
const PAST = 0.4
/**
 * The radians in one degree.
 */
const RAD = Math.PI / 180
/**
 * How the ring of each stop on the dial's rim is drawn.
 */
const RIM = { 'fill': markup.TONE.ground, 'r': 7, 'stroke': markup.TONE.lead, 'stroke-width': 1.5 } as const
/**
 * How far right of an angle a reading's name starts where the two share a
 * line, the room an angle in the reading's 21 px face takes.
 */
const SHARE = 86

/**
 * Draws the rule at `row`'s reading beside the dial into `plot`: its name, the
 * angle and the share its farthest drones turn through, and the share's bar
 * over its track with the whisker of its standard error, the bar running the
 * readings' column. A wide or narrow chart sets the name on a line over the
 * angle and the share, and one between sets the shorter name beside the
 * angle.
 */
function drawReading(
  layout : Layout,
  plot   : SVGElement,
  row    : number,
  rule   : Rule
): Omit<Ring, 'arc' | 'band' | 'groups' | 'pastArc' | 'pastBand'> & { group: SVGElement } {
  const { center, narrow, readTo, readX, tall, wide } = layout
  const beside  = !narrow && !wide
  const top     = narrow ? center[1] + 26 + row * tall : wide ? 3 + row * tall : row * tall
  const line    = top + (beside ? 21 : 35)
  const barY    = line + 6
  const length  = readTo - readX
  const angle   = markup.text(markup.FACE.mono, 21, '', readX, line, { 'font-weight': 500 })
  const muted   = { 'fill': markup.TONE.muted, 'text-anchor': 'end' }
  const share   = markup.text(markup.FACE.mono, 11.5, '', readTo, line, muted)
  const bar     = markup.svg('rect', { fill: rule.color, height: 4, rx: 2, width: 0, x: readX, y: barY })
  const track   = { fill: rule.color, height: 4, opacity: 0.15, rx: 2, width: length, x: readX, y: barY }
  const tick    = { 'opacity': 0.5, 'stroke': markup.TONE.ink, 'stroke-width': 1 }
  const whisker = markup.svg('line', { ...tick, y1: barY + 2, y2: barY + 2 })
  const named   = { 'fill': rule.color, 'font-weight': row === 0 ? 600 : 400 }
  const group   = markup.svg('g', { class: 'rules-fade' },
    markup.svg('rect', { fill: 'transparent', height: tall - 4, width: length + 6, x: readX - 6, y: top }),
    beside
      ? markup.text(markup.FACE.body, 14, rule.short, readX + SHARE, line, named)
      : markup.text(markup.FACE.body, 14, wide ? rule.label : rule.short, readX, top + 11, named),
    angle,
    share,
    markup.svg('rect', track),
    bar,
    whisker)
  plot.append(group)
  return { angle, bar, group, share, whisker, width: length }
}

/**
 * Finds how far the dial at `outer` reaches left and right of its center: the
 * rim past the vertical out to `reach` on the left, and the rim or the label
 * of the lowest of `turns` on the right.
 */
function extent(outer: number, reach: number, turns: readonly number[]): { left: number, right: number } {
  return {
    left  : Math.min(0, (outer + 6) * Math.cos(reach * RAD)) - 2,
    right : Math.max(outer + 6, ...turns.map(turn => (outer + 24) * Math.cos(turn * RAD) + 14))
  }
}

/**
 * Finds the widest outer radius, up to `cap`, whose dial spans no more than
 * `span`.
 */
function outerFor(cap: number, reach: number, span: number, turns: readonly number[]): number {
  let outer = cap
  while (outer > 80) {
    const { left, right } = extent(outer, reach, turns)
    if (right - left <= span) break
    outer -= 1
  }
  return outer
}

/**
 * Finds the widest angle any rule's ring or band reaches over every turn,
 * rounded up to a multiple of 15°, never under the quarter.
 */
const reachOf = (rules: readonly Rule[]): number => Math.max(90, Math.ceil(Math.max(...rules.flatMap(rule =>
  rule.points.map(point => (point.mean + point.error) * point.given))) / 15) * 15)

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
  const angle = degrees * RAD
  return [layout.center[0] + radius * Math.cos(angle), layout.center[1] - radius * Math.sin(angle)]
}

/**
 * Draws the dial's frame into `plot`, a spoke every 30° out to the frame's
 * reach under its rim and its caption, and returns the dashed marker of the
 * first drone's turn.
 */
export function drawFrame(layout: Layout, plot: SVGElement): SVGElement {
  const { center, outer, radii, reach, ring } = layout
  for (let degrees = 0; degrees <= reach; degrees += 30) {
    const [x1, y1] = at(layout, radii.at(-1)! - ring / 2 - 4, degrees)
    const [x2, y2] = at(layout, outer + 2, degrees)
    plot.append(markup.svg('line', { 'stroke': markup.TONE.rule, 'stroke-width': 1, x1, x2, y1, y2 }))
  }
  const rim = { 'fill': 'none', 'stroke': markup.TONE.rule, 'stroke-width': 1 }
  plot.append(
    markup.svg('path', { ...rim, d: arc(layout, outer + 6, 0, reach) }),
    markup.text(markup.FACE.mono, 10.5, 'THE FIRST DRONE TURNS', center[0] + 26, 14, markup.CAPTION)
  )
  return markup.svg('path', {
    'fill'             : 'none',
    'stroke'           : markup.TONE.ink,
    'stroke-dasharray' : '5 4',
    'stroke-width'     : 1.5
  })
}

/**
 * Draws the disc in the page's ground color under the stop for the turn of
 * `turn` degrees on the dial's rim.
 */
export function drawHalo(layout: Layout, turn: number): SVGElement {
  const [cx, cy] = at(layout, layout.outer + 6, turn)
  return markup.svg('circle', { cx, cy, fill: markup.TONE.ground, r: HALO })
}

/**
 * Draws the stop for the turn of `turn` degrees on the dial's rim, whose dot
 * shows while it is picked, out of a screen reader's hearing, since the radio
 * standing over it carries its name.
 */
export function drawKnob(layout: Layout, turn: number): SVGElement {
  const [x, y]   = at(layout, layout.outer + 6, turn)
  const [lx, ly] = at(layout, layout.outer + 24, turn)
  return markup.svg('g', { 'aria-hidden': 'true', 'class': 'rules-knob' },
    markup.svg('circle', { ...RIM, cx: x, cy: y }),
    markup.svg('circle', { class: 'rules-knob-dot', cx: x, cy: y, fill: markup.TONE.lead, r: 3.5 }),
    markup.text(markup.FACE.mono, 11.5, `${turn}°`, lx, ly + 4, { 'text-anchor': 'middle' }))
}

/**
 * Draws the rule at `row` into `plot`: its ring on the dial over a faint
 * quarter, the band of its standard error along the ring's edge, the lighter
 * ring and band past the first drone's turn, and its reading beside the dial.
 */
export function drawRing(layout: Layout, plot: SVGElement, row: number, rule: Rule): Ring {
  const stroke              = { 'fill': 'none', 'stroke': rule.color, 'stroke-width': layout.ring }
  const path                = markup.svg('path', stroke)
  const band                = markup.svg('path', { ...stroke, ...BAND })
  const pastArc             = markup.svg('path', { ...stroke, opacity: PAST })
  const pastBand            = markup.svg('path', { ...stroke, ...BAND, opacity: PAST * BAND.opacity })
  const quarter             = arc(layout, layout.radii[row]!, 0, 90)
  const backing             = markup.svg('path', { ...stroke, d: quarter, opacity: 0.1 })
  const arcs                = markup.svg('g', { class: 'rules-fade' }, backing, path, band, pastArc, pastBand)
  plot.append(arcs)
  const { group, ...shown } = drawReading(layout, plot, row, rule)
  return { ...shown, arc: path, band, groups: [arcs, group], pastArc, pastBand }
}

/**
 * Lays the chart out at `width` for `rules`: under the narrow width the
 * readings sit under a dial as wide as the chart, and from it on the dial and
 * the readings each take one of two equal columns, the readings spread down
 * the dial's height and ending on their column's right edge.
 */
export function layoutOf(rules: readonly Rule[], tuning: Tuning, width: number): Layout {
  const reach  = reachOf(rules)
  const turns  = rules[0]!.points.map(point => point.given)
  const narrow = width < tuning.narrow
  const wide   = !narrow && width >= tuning.wide
  const gutter = wide ? 56 : 40
  const outer  = narrow
    ? outerFor(250, reach, width, turns)
    : outerFor(330, reach, Math.floor((width - gutter) / 2), turns)
  const base   = outer + 34
  const ring   = Math.min(24, outer / 7)
  const tall   = narrow ? 58 : (base - (wide ? 48 : 31)) / (rules.length - 1)
  const { left, right } = extent(outer, reach, turns)
  const readX  = narrow ? 0 : Math.round(right - left) + gutter
  return {
    center : [Math.round(-left), base],
    height : narrow ? base + 26 + rules.length * tall : base + 8,
    narrow : narrow,
    outer  : outer,
    radii  : Array.from({ length: rules.length }, (_, row) => outer - row * (ring + 8) - ring / 2),
    reach  : reach,
    readTo : narrow ? width : 2 * readX - gutter,
    readX  : readX,
    ring   : ring,
    tall   : tall,
    wide   : wide,
    width  : width
  }
}
