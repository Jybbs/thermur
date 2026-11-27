import { type Box, lerp, mulberry, smoothstep } from '../common.ts'
import { seriesProperty }                       from '../identity.ts'

/**
 * One source's number: its numeral's box, and whether its source is current.
 */
interface Anchor {
  box     : Box
  current : boolean
}

/**
 * One bird of the flock: traced bird `glyph` of the mark with its head at `x`
 * and `y`, turned to `angle` and `length` pixels long, colored at `t` along
 * the mark's gradient, or in the lead hue where it labels the current source.
 */
export interface Bird {
  angle    : number
  glyph    : number
  length   : number
  t        : number
  x        : number
  y        : number
  current? : boolean
}

/**
 * A circle of radius `r` centered at `x` and `y`.
 */
interface Circle {
  r : number
  x : number
  y : number
}

/**
 * The tuning of a layout, as `LAYOUT` declares it.
 */
interface Layout {
  gap   : number
  pitch : number
  seed  : number
}

/**
 * What the flock is laid out around, in the Sources section's own pixels: each
 * source's number, the column the numbers stand in from the first source to the
 * last, the heading's row and its text, and the section's width.
 */
export interface Scene {
  anchors : readonly Anchor[]
  rail    : Box
  row     : Box
  title   : Box
  width   : number
}

/**
 * A point of the line the flock follows, with its share of the way along it.
 */
interface Spine {
  along : number
  x     : number
  y     : number
}

/**
 * The traced bird of the mark each number labels.
 */
const LABEL = 26
/**
 * How a layout spreads the flock: the pixels it stops short of the section's
 * right edge, the pixels between two points of its line up the column, which
 * set its density there, and the seed it draws from, so a page at one width
 * lays out alike on every load.
 */
const LAYOUT: Layout = { gap: 72, pitch: 15, seed: 23 }

/**
 * The bird a source's number labels, its head just short of the numeral and
 * turned up to the right as the mark's birds are, colored at `t`.
 */
const labeled = ({ box, current }: Anchor, t: number): Bird =>
  ({ angle: -0.5, current, glyph: LABEL, length: 15, t, x: box.x0 - 3, y: lerp(box.y0, box.y1, 0.4) })

/**
 * The center and reach of a bird's body, the circle a layout keeps clear of
 * the numbers, the heading's text, and the other birds.
 */
export const body = ({ angle, length, x, y }: Bird): Circle =>
  ({ r: length * 0.42, x: x - Math.cos(angle) * length * 0.55, y: y - Math.sin(angle) * length * 0.55 })

/**
 * Whether two birds' bodies come within `pad` pixels of each other.
 */
export function crowds(a: Bird, b: Bird, pad: number): boolean {
  const [p, q] = [body(a), body(b)]
  return Math.hypot(p.x - q.x, p.y - q.y) < p.r + q.r + pad
}

/**
 * The fill of a bird: the lead hue where it labels the current source, and
 * otherwise its place along the mark's gradient, from the ink at the flock's
 * foot to Fire Lord at its head, eased as the mark eases it.
 */
export const fillOf = ({ current, t }: Bird): string => current
  ? 'var(--th-lead)'
  : `color-mix(in srgb, var(${seriesProperty(0)}) ${(smoothstep(0, 0.8, t) * 100).toFixed(1)}%, `
  + 'var(--th-ink))'

/**
 * Lays the flock out over `scene` from `glyphs` traced birds, rising up the
 * column of numbers from the last source to the first and turning right along
 * the heading's row past its text, as the mark's flock rises and turns. Each
 * number labels a bird whose head sits just short of it, and every other bird
 * keeps clear of the numbers, the heading's text, the entries, and the other
 * birds, or is left out.
 */
export function flock(scene: Scene, glyphs: number, layout: Layout = LAYOUT): Bird[] {
  const { anchors, rail, row, title, width } = scene
  const draw    = mulberry(layout.seed)
  const end     = width - layout.gap
  const line    = spine(rail, end, (row.y0 + title.y1) / 2 + 4, layout)
  const nearest = (y: number): Spine =>
    line.reduce((best, each) => Math.abs(each.y - y) < Math.abs(best.y - y) ? each : best)
  const numbered      = anchors.map(anchor => labeled(anchor, 0.1 + 0.9 * nearest(anchor.box.y0).along))
  const heading       = { ...title, x1: title.x1 + 8 }
  const loose: Bird[] = []
  for (let tries = 0; tries < line.length * 12 && loose.length < line.length * 1.2; tries++) {
    const place  = Math.floor(draw() * line.length)
    const point  = line[place]!
    const ahead  = line[Math.min(line.length - 1, place + 1)]!
    const angle  = Math.atan2(ahead.y - point.y, ahead.x - point.x) + (draw() - 0.5) * 0.5
    const spread = (draw() + draw() + draw() - 1.5) * (14 - 6 * point.along)
    const bird   = {
      angle  : angle,
      glyph  : Math.floor(draw() * glyphs),
      length : 8 + 7 * point.along + draw() * 2,
      t      : 0.1 + 0.9 * point.along + (draw() - 0.5) * 0.1,
      x      : point.x - Math.sin(angle) * spread,
      y      : point.y + Math.cos(angle) * spread
    }
    const reach       = body(bird)
    const { r, x, y } = reach
    const inRail      = x - r >= rail.x0 + 2 && x + r <= rail.x1 - 2 && y + r <= rail.y1
    const inRow       = y - r >= row.y0 + 2 && y + r <= rail.y0 - 4 && x + r <= end + 16
    const free        = !touches(reach, heading, 4) && loose.every(other => !crowds(bird, other, 1))
    if ((inRail || inRow) && free) loose.push(bird)
  }
  const clear = loose.filter(each => {
    const reach = body(each)
    return anchors.every(anchor => !touches(reach, anchor.box, 3))
        && numbered.every(other => !crowds(each, other, 3))
  })
  return [...numbered, ...clear]
}

/**
 * Samples the line the flock follows, one point every `layout.pitch` pixels up
 * the column `rail` from the last source to the first, then bending right into
 * the heading's row at `row` and running along it to `end`, each point
 * carrying its share of the way along the line.
 */
export function spine(rail: Box, end: number, row: number, layout: Layout = LAYOUT): Spine[] {
  const x      = lerp(rail.x0, rail.x1, 0.45)
  const bend   = Math.min(80, (rail.y1 - row) * 0.3)
  const rise   = Math.max(40, Math.ceil((rail.y1 - 8 - row - bend) / layout.pitch))
  const points = [
    ...Array.from({ length: rise }, (_, k) =>
      ({ x: lerp(x - 6, x, k / (rise - 1)), y: lerp(rail.y1 - 8, row + bend, k / (rise - 1)) })),
    ...Array.from({ length: 24 }, (_, k) => {
      const s = (k + 1) / 24
      return { x: lerp(x, end, 0.6 * s * s), y: row + bend * (1 - s) * (1 - s) }
    }),
    ...Array.from({ length: 16 }, (_, k) => ({ x: lerp(x, end, 0.6 + 0.4 * (k + 1) / 16), y: row }))
  ]
  const steps = points.map((each, k) => {
    const last = points[k - 1] ?? each
    return Math.hypot(each.x - last.x, each.y - last.y)
  })
  const total = steps.reduce((sum, each) => sum + each, 0)
  let walked  = 0
  return points.map(({ x: across, y: down }, k) =>
    ({ along: (walked += steps[k]!) / total, x: across, y: down }))
}

/**
 * Whether the circle at `x` and `y` of radius `r` comes within `pad` pixels of
 * `box`.
 */
export function touches({ r, x, y }: Circle, box: Box, pad: number): boolean {
  const dx = Math.max(box.x0 - pad - x, 0, x - box.x1 - pad)
  const dy = Math.max(box.y0 - pad - y, 0, y - box.y1 - pad)
  return dx * dx + dy * dy < r * r
}
