/**
 * The attributes an element of a drawing takes, any left undefined left unset.
 */
export type Attributes = Record<string, number | string | undefined>

/**
 * The faces a drawing sets its words in, read from the custom properties the
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
 * The SVG namespace every element of a drawing is made in.
 */
const SVG = 'http://www.w3.org/2000/svg'
/**
 * The colors a drawing paints in, each a custom property the identity's
 * stylesheet sets, so the drawing follows the site's appearance switch.
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

/**
 * Calls `changed` with the host's width, rounded to a pixel, whenever the host
 * takes a nonzero width other than the one it was last drawn at, returning
 * what releases the observer.
 */
export function widthOf(host: HTMLElement, changed: (width: number) => void): () => void {
  let drawn      = 0
  const observer = new ResizeObserver(() => {
    const width = Math.round(host.getBoundingClientRect().width)
    if (width === drawn || width === 0) return
    drawn = width
    changed(width)
  })
  observer.observe(host)
  return () => observer.disconnect()
}

/**
 * Where a control laid over a drawing stands, in pixels from the drawing's top
 * left corner: the point it centers on, and for a control covering a stretch of
 * the drawing, how wide and tall that stretch is.
 */
export interface Spot {
  height? : number
  left    : number
  top     : number
  width?  : number
}

/**
 * Writes `length` in pixels, or nothing where it is unset.
 */
const px = (length: number | undefined): string | undefined => length === undefined ? undefined : `${length}px`

/**
 * Writes `spot` as the style placing a control over a drawing, any size left
 * unset left to the control's stylesheet.
 */
export const placed = ({ height, left, top, width }: Spot): Record<string, string | undefined> => ({
  height : px(height),
  left   : px(left),
  top    : px(top),
  width  : px(width)
})
