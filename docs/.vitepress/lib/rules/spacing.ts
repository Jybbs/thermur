import { markup } from '@thermur/kit'

import * as chart    from './chart.ts'
import type { Rule } from './measured.ts'

/**
 * Where the coupling chart sets each of its parts at one width: whether it
 * lays itself out narrow, its height, the column the names end at, the
 * polarization axis's ends, each row's height, the top of the first row, the
 * track's height, and the maps from a spacing and from a polarization to a
 * pixel across the chart.
 */
export interface Layout {
  fromS  : (spacing: number) => number
  height : number
  label  : number
  left   : number
  narrow : boolean
  right  : number
  row    : number
  toX    : (polarization: number) => number
  top    : number
  ty     : number
  width  : number
}
/**
 * What the track drawn into the chart holds: the share of the groove behind
 * the handle, the glyph of the knob that plays the sweep, the handle, the
 * spacing read above the handle, and where the knob and the groove stand, so
 * the controls that play the sweep and move the handle can stand over them.
 */
export interface Rail {
  fill    : SVGElement
  glyph   : SVGElement
  handle  : SVGElement
  player  : markup.Spot
  readout : SVGElement
  track   : markup.Spot
}
/**
 * What one rule's row holds once drawn: its dot, the dashed tie from the dot
 * to the default's value, its reading at the row's end, and the group the row
 * lifts through.
 */
export interface Row {
  dot   : SVGElement
  group : SVGElement
  tie   : SVGElement
  value : SVGElement
}
/**
 * How the chart lays itself out and moves: the narrowest width it sets its
 * names beside the rows at, and the seconds one sweep of the track takes from
 * the densest launch to the sparsest and back.
 */
export interface Tuning {
  narrow : number
  sweep  : number
}

/**
 * How the track's caption is set.
 */
const CAPTION = { 'fill': markup.TONE.muted, 'letter-spacing': '0.08em' } as const
/**
 * How each rule's dot is ringed against the ground.
 */
const DOT = { 'stroke': markup.TONE.ground, 'stroke-width': 2 } as const
/**
 * How words set to end at their place are anchored.
 */
const END = { 'text-anchor': 'end' } as const
/**
 * The play and pause glyphs on the track's knob, as paths around its center.
 */
export const GLYPH = {
  pause : 'M-4,-5 L-1,-5 L-1,5 L-4,5 Z M1,-5 L4,-5 L4,5 L1,5 Z',
  play  : 'M-3.5,-5 L5,0 L-3.5,5 Z'
} as const
/**
 * How the track's groove is stroked.
 */
const GROOVE = { 'stroke-linecap': 'round', 'stroke-width': 4 } as const
/**
 * How each line of the axis's grid is stroked.
 */
const HAIRLINE = { 'stroke': markup.TONE.rule, 'stroke-width': 1 } as const
/**
 * How words set to center on their place are anchored.
 */
export const MIDDLE = { 'text-anchor': 'middle' } as const
/**
 * How the words in the chart's muted tone are set.
 */
export const MUTED = { fill: markup.TONE.muted } as const
/**
 * The name each coupling rule reads under, the default's first, beside the
 * shorter name a narrow chart sets.
 */
export const NAMES = {
  reference   : ['Constant, the default', 'Constant'],
  anisotropic : ['Anisotropic', 'Anisotropic'],
  weighted    : ['Distance-weighted', 'Weighted'],
  metric      : ['Metric', 'Metric']
} as const
/**
 * How each measured spacing's stop on the track is drawn.
 */
const STOP = { 'fill': markup.TONE.ground, 'r': 3, 'stroke': markup.TONE.muted, 'stroke-width': 1.25 } as const
/**
 * The polarizations the chart marks along its axis.
 */
const TICKS = [0, 0.25, 0.5, 0.75, 1] as const
/**
 * How the dashed tie from each dot to the default's value is stroked.
 */
const TIE = { 'stroke': markup.TONE.faint, 'stroke-dasharray': '2 3', 'stroke-width': 1 } as const
/**
 * How the chart lays itself out and moves unless told otherwise.
 */
export const TRACK: Tuning = { narrow: 520, sweep: 7 }

/**
 * Draws the track's handle at `spacing` into `plot`, beside the strip a press
 * lands on and the spacing read above the handle.
 */
function drawHandle(
  layout  : Layout,
  plot    : SVGElement,
  spacing : number,
  stops   : readonly number[]
): Pick<Rail, 'handle' | 'readout' | 'track'> {
  const { fromS, ty } = layout
  const [lo, hi]      = [stops[0]!, stops.at(-1)!]
  const bold          = { ...MIDDLE, 'font-weight': 600 }
  const readout       = markup.text(markup.FACE.mono, 11.5, '', fromS(spacing), ty - 14, bold)
  const handle        = markup.svg('circle', {
    'class'        : 'rules-handle',
    'cy'           : ty,
    'fill'         : markup.TONE.lead,
    'r'            : 8,
    'stroke'       : markup.TONE.ground,
    'stroke-width' : 2
  })
  const track         = { height: 28, left: fromS(lo), top: ty - 14, width: fromS(hi) - fromS(lo) }
  plot.append(readout, handle)
  return { handle, readout, track }
}

/**
 * Draws the polarization axis under the rows into `plot`, its ticks and its
 * name, and returns the dashed guide standing at the default's value.
 */
export function drawAxis(layout: Layout, plot: SVGElement): SVGElement {
  const { height, toX, top, width } = layout
  for (const tick of TICKS) {
    const anchor = tick === 0 ? 'start' : tick === 1 ? 'end' : 'middle'
    const x      = toX(tick)
    plot.append(
      markup.svg('line', { ...HAIRLINE, x1: x, x2: x, y1: top, y2: height - 26 }),
      markup.text(markup.FACE.mono, 10.5, tick.toFixed(2), x, height - 12, { ...MUTED, 'text-anchor': anchor })
    )
  }
  const guide = markup.svg('line', {
    'opacity'          : 0.6,
    'stroke'           : chart.series(0),
    'stroke-dasharray' : '3 3',
    'stroke-width'     : 1,
    'y1'               : top,
    'y2'               : height - 26
  })
  const named = markup.text(markup.FACE.body, 12.5, 'Polarization', width - 4, height - 12, {
    ...MUTED,
    ...END
  })
  plot.append(named, guide)
  return guide
}

/**
 * Draws the track along the chart's top into `plot`: its caption, the knob
 * playing the sweep, the groove with a stop at each measured spacing, and the
 * handle at `spacing`, the sweep `playing` or not.
 */
export function drawRail(
  layout  : Layout,
  playing : boolean,
  plot    : SVGElement,
  spacing : number,
  stops   : readonly number[]
): Rail {
  const { fromS, left, narrow, ty } = layout
  const [lo, hi]                    = [stops[0]!, stops.at(-1)!]
  const caption                     = narrow
    ? markup.text(markup.FACE.mono, 10.5, 'LAUNCH SPACING', left, 12, CAPTION)
    : markup.text(markup.FACE.mono, 10.5, 'LAUNCH SPACING', left - 44, ty + 4, { ...CAPTION, ...END })
  const glyph                       = markup.svg('path', {
    d    : playing ? GLYPH.pause : GLYPH.play,
    fill : markup.TONE.lead
  })
  const player                      = { left: narrow ? left + 11 : left - 22, top: ty }
  const knob                        = markup.svg('g', { transform: `translate(${player.left} ${player.top})` },
    markup.svg('circle', { 'fill': 'none', 'r': 11, 'stroke': markup.TONE.lead, 'stroke-width': 1.25 }), glyph)
  const groove                      = { ...GROOVE, x1: fromS(lo), y1: ty, y2: ty }
  const filled                      = { opacity: 0.55, stroke: markup.TONE.lead, x2: fromS(spacing) }
  const fill                        = markup.svg('line', { ...groove, ...filled })
  plot.append(caption, knob, markup.svg('line', { ...groove, stroke: markup.TONE.rule, x2: fromS(hi) }), fill)
  for (const each of stops) {
    plot.append(
      markup.svg('circle', { ...STOP, cx: fromS(each), cy: ty }),
      markup.text(markup.FACE.mono, 10.5, `${each}`, fromS(each), ty + 17, { ...MUTED, ...MIDDLE })
    )
  }
  return { fill, glyph, player, ...drawHandle(layout, plot, spacing, stops) }
}

/**
 * Draws the row of the rule at `place` into `plot`: its name, the band its
 * measured values span with a dot at each, and the dot, the tie, and the
 * reading the chart moves.
 */
export function drawRow(layout: Layout, place: number, plot: SVGElement, rule: Rule): Row {
  const { label, left, narrow, row, toX, top, width } = layout
  const named  = { 'fill': rule.color, 'font-weight': place === 0 ? 600 : 400 }
  const cy     = top + place * row + (narrow ? 34 : row / 2)
  const all    = rule.points.map(each => each.mean)
  const group  = markup.svg('g', { class: 'rules-fade' })
  const tie    = markup.svg('line', { ...TIE, y1: cy, y2: cy })
  const dot    = markup.svg('circle', { ...DOT, cy, fill: rule.color, r: place === 0 ? 8 : 7 })
  const value  = markup.text(markup.FACE.mono, 12, '', width - 4, cy + 4, END)
  const name   = narrow
    ? markup.text(markup.FACE.body, 13.5, rule.label, left, cy - 14, named)
    : markup.text(markup.FACE.body, 15, rule.label, label, cy + 5, { ...named, ...END })
  const span   = markup.svg('line', {
    'opacity'        : 0.16,
    'stroke'         : rule.color,
    'stroke-linecap' : 'round',
    'stroke-width'   : 6,
    'x1'             : toX(Math.min(...all)),
    'x2'             : toX(Math.max(...all)),
    'y1'             : cy,
    'y2'             : cy
  })
  group.append(
    markup.svg('rect', { fill: 'transparent', height: row, width, x: 0, y: cy - row / 2 - (narrow ? 8 : 0) }),
    name,
    span,
    ...rule.points.map(each =>
      markup.svg('circle', { cx: toX(each.mean), cy, fill: rule.color, opacity: 0.55, r: 2.25 })),
    tie,
    dot,
    value
  )
  plot.append(group)
  return { dot, group, tie, value }
}

/**
 * Lays the chart out at `width` for `count` rules over the spacings `lo` to
 * `hi`.
 */
export function layoutOf(count: number, hi: number, lo: number, tuning: Tuning, width: number): Layout {
  const narrow = width < tuning.narrow
  const label  = narrow ? 0 : 168
  const left   = label + (narrow ? 8 : 16)
  const right  = width - (narrow ? 84 : 100) - 8
  const start  = left + (narrow ? 34 : 6)
  const row    = narrow ? 50 : 38
  const top    = narrow ? 68 : 54
  return {
    fromS  : chart.linear([lo, hi], [start, right]),
    height : top + count * row + 30,
    label,
    left,
    narrow,
    right,
    row,
    toX    : chart.linear([0, 1], [left, right]),
    top,
    ty     : narrow ? 42 : 24,
    width
  }
}
