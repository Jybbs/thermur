import { seriesProperty } from '../identity.ts'

/**
 * The attributes an element of a chart takes, any left undefined left unset.
 */
export type Attributes = Record<string, number | string | undefined>

/**
 * The faces a chart sets its words in, read from the custom properties the
 * identity's stylesheet sets: the body's face and the code's.
 */
export const FACE = {
  body : 'var(--vp-font-family-base)',
  mono : 'var(--vp-font-family-mono)'
} as const
/**
 * The attributes set as style properties, since a custom property resolves in a
 * style in every engine and in a presentation attribute in only some.
 */
const STYLED = new Set(['fill', 'font-family', 'opacity', 'stroke'])
/**
 * The SVG namespace every element of a chart is made in.
 */
const SVG = 'http://www.w3.org/2000/svg'
/**
 * The colors a chart draws in, each a custom property the identity's
 * stylesheet sets, so the chart follows the site's appearance switch.
 */
export const TONE = {
  faint  : 'var(--vp-c-text-3)',
  ground : 'var(--vp-c-bg)',
  ink    : 'var(--vp-c-text-1)',
  lead   : 'var(--th-lead)',
  muted  : 'var(--vp-c-text-2)',
  rule   : 'var(--vp-c-divider)'
} as const

/**
 * Maps the span `from` onto the span `to` along a straight line.
 */
export const linear = (from: readonly [number, number], to: readonly [number, number]) =>
  (value: number): number => to[0] + (value - from[0]) / (from[1] - from[0]) * (to[1] - to[0])

/**
 * Names the color of the chart series at `place`, the fire's at place 0.
 */
export const series = (place: number): string => `var(${seriesProperty(place)})`

/**
 * Sets `attributes` on `node`, each color, face, and opacity as a style
 * property.
 */
export function set(node: Element, attributes: Attributes): void {
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined) continue
    if (STYLED.has(key)) (node as SVGElement).style.setProperty(key, String(value))
    else node.setAttribute(key, String(value))
  }
}

/**
 * Makes one SVG element of `tag` with `attributes` set and `kids` appended.
 */
export function svg(tag: string, attributes: Attributes = {}, ...kids: (Node | string)[]): SVGElement {
  const node = document.createElementNS(SVG, tag) as SVGElement
  set(node, attributes)
  node.append(...kids)
  return node
}

/**
 * Makes a line of `words` at `x` and `y` in `face` at `size` pixels, in the
 * ink unless `more` sets another fill.
 */
export const text = (
  face  : string,
  size  : number,
  words : string,
  x     : number,
  y     : number,
  more  : Attributes = {}
): SVGElement =>
  svg('text', { 'fill': TONE.ink, 'font-family': face, 'font-size': size, x, y, ...more }, words)
