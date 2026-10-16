import { type Pair, mulberry } from './common.ts'

/**
 * A sky's bands, the band at index `k` filled with `tones[k]` between
 * `edges[k]` and `edges[k + 1]` across `width` pixels, each edge raised by what
 * `lift` returns where a plume rises through it.
 */
interface Bands {
  edges : readonly number[]
  lift  : (x: number, y: number) => number
  tones : readonly string[]
  width : number
}

/**
 * A traced bird a scene draws: its head at `x` and `y`, turned `angle` radians,
 * `size` pixels long, drawing the traced bird at index `glyph` in the color
 * `fill`.
 */
interface Bird {
  angle : number
  fill  : string
  glyph : number
  size  : number
  x     : number
  y     : number
}

/**
 * A murmuration as `murmuration` lays it out: `count` birds scattered across a
 * ribbon that bends along `spine` and spreads to `width` either side of it,
 * drawn from `seed`, each as long as `size` returns for its place along the
 * ribbon and a draw in [0, 1), each drawing one of the `glyphs` traced birds in
 * the color `color` returns, and each left out or turned aside where `gap`
 * reports the heat it would sit in. Where `density` is given, a bird at each
 * place along the ribbon is kept at the share it returns there.
 */
interface Murmuration {
  color    : (along: number, edge: number, x: number, y: number) => string
  count    : number
  gap      : (x: number, y: number) => number
  glyphs   : number
  seed     : number
  size     : (along: number, draw: number) => number
  spine    : (along: number) => Pair
  width    : (along: number) => number
  density? : (along: number) => number
}

/**
 * Lists the points from 0 to `x1` every `step` pixels.
 */
const steps = (x1: number, step: number): number[] =>
  Array.from({ length: Math.floor(x1 / step) + 1 }, (_, index) => index * step)

/**
 * Writes the blur filter `id` as SVG, a Gaussian blur with a standard deviation
 * of `deviation` pixels over a region twice the width of the shape it blurs and
 * `height` percent of its height, centered on it.
 */
export const blur = (id: string, deviation: number, height: number): string =>
    `<filter id="${id}" x="-50%" y="${(100 - height) / 2}%" width="200%" height="${height}%">`
  + `<feGaussianBlur stdDeviation="${deviation}"/></filter>`

/**
 * Measures how deep in a heat a point `reach` from it sits, 2 inside `edge`,
 * where no bird flies, and outside it a share falling from `fade - edge` at the
 * edge to nothing at `fade`.
 */
export const depth = (reach: number, edge: number, fade: number): number =>
  reach < edge ? 2 : Math.max(0, fade - reach)

/**
 * Draws the definition of each traced bird a scene uses, the bird at index `i`
 * of `glyphs` under the id `g<i>`, each a unit polygon with its head at the
 * origin facing along the x axis and its tail at x = −1.
 */
export const glyphDefs = (glyphs: readonly string[]): string =>
  glyphs.map((d, index) => `<path id="g${index}" d="${d}"/>`).join('')

/**
 * Reads the path data of each traced bird a mark's SVG draws, in the order the
 * mark draws them.
 */
export const glyphsOf = (svg: string): string[] =>
  [...svg.matchAll(/<path\b[^>]*?\sd="([^"]+)"/gu)].map(([, d]) => d!)

/**
 * Lays a murmuration out as traced birds along a bent ribbon, drawing up to
 * forty candidates per bird. A candidate where `gap` reports more than one sits
 * inside the heat and is left out, and one past 0.35 sits at the edge of the
 * parting, turned away from it and lifted, as `parted` places it.
 */
export function murmuration(flock: Murmuration): string {
  const draw            = mulberry(flock.seed)
  const density         = flock.density ?? ((): number => 1)
  const birds: string[] = []
  for (let tries = 0; birds.length < flock.count && tries < flock.count * 40; tries++) {
    const along = draw()
    if (draw() > density(along)) continue
    const [sx, sy] = flock.spine(along)
    const [ax, ay] = flock.spine(Math.min(1, along + 0.01))
    const length   = Math.hypot(ax - sx, ay - sy) || 1
    const [tx, ty] = [(ax - sx) / length, (ay - sy) / length]
    const offset   = (draw() + draw() + draw() - 1.5) / 1.5 * flock.width(along)
    const slide    = (draw() - 0.5) * 18
    const x        = sx - ty * offset + tx * slide
    const y        = sy + tx * offset + ty * slide
    const heat     = flock.gap(x, y)
    if (heat > 1) continue
    const edge  = Math.max(0, heat)
    const angle = Math.atan2(ty, tx) + (draw() - 0.5) * 0.5
    const glyph = Math.floor(draw() * flock.glyphs)
    const fill  = flock.color(along, edge, x, y)
    birds.push(parted({ angle, fill, glyph, size: flock.size(along, draw()), x, y }, edge))
  }
  return birds.join('')
}

/**
 * Writes the bird as an SVG `use` of its traced bird's definition, turned `0.8
 * × edge` radians away from the parting and lifted `6 × edge` pixels where
 * `edge`, how deep in the heat it sits, passes 0.35.
 */
export function parted({ angle, fill, glyph, size, x, y }: Bird, edge: number): string {
  const turn  = edge > 0.35 ? edge : 0
  const place = `translate(${tenths(x)} ${tenths(y - 6 * turn)}) `
              + `rotate(${tenths((angle - 0.8 * turn) * 180 / Math.PI)}) scale(${tenths(size)})`
  return `<use href="#g${glyph}" fill="${fill}" transform="${place}"/>`
}

/**
 * Writes `points` as the coordinates of an SVG path, each rounded to tenths and
 * joined by line commands, with no opening move.
 */
export const polyline = (points: readonly Pair[]): string =>
  points.map(([x, y]) => `${tenths(x)} ${tenths(y)}`).join('L')

/**
 * Draws a sky's bands as SVG paths, each edge traced every eight pixels.
 */
export function skyBands({ edges, lift, tones, width }: Bands): string {
  const xs   = steps(width, 8)
  const edge = (y: number, across: readonly number[]): string =>
    polyline(across.map(x => [x, y - lift(x, y)]))
  return tones.map((tone, k) =>
    `<path d="M${edge(edges[k]!, xs)}L${edge(edges[k + 1]!, xs.toReversed())}Z" fill="${tone}"/>`).join('')
}

/**
 * Rounds `value` to tenths, the precision a scene writes its coordinates at.
 */
export const tenths = (value: number): number => Math.round(value * 10) / 10

/**
 * Traces `height` across a card from its left edge to `x1` as an SVG path, one
 * point every six pixels.
 */
export const trace = (x1: number, height: (x: number) => number): string =>
  `M${polyline(steps(x1, 6).map(x => [x, height(x)]))}`
