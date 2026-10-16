import * as common         from './common.ts'
import type { Curve }      from './curve.ts'
import type { Fireground } from './fireground.ts'
import { CANVAS }          from './layout.ts'
import * as scene          from './scene.ts'

/**
 * How a flock is drawn at each point along its route, by pixels from the
 * wordmark's tip: its half width, its birds to a pixel of route, each bird's
 * length for a draw in [0, 1), and each bird's color at a point on the card for
 * how deep in the heat it sits.
 */
interface Profile {
  color : (s: number, edge: number, x: number, y: number) => string
  perPx : (s: number) => number
  size  : (s: number, draw: number) => number
  width : (s: number) => number
}

/**
 * What the wake is drawn from: the route it flies from the wordmark's tip into
 * its ring around the fire, the boxes the words take, the fire it parts around,
 * the number of traced birds it draws from, the two colors it mixes, cream and
 * the fire's own, and the wordmark's trailing chevrons, each `chevron` pixels
 * long in rows `rows` pixels apart.
 */
interface Wake {
  boxes   : readonly common.Box[]
  chevron : number
  cream   : string
  fire    : Fireground
  glyphs  : number
  lift    : string
  route   : Curve
  rows    : number
}

/**
 * How the wake is drawn, in pixels along its route from the wordmark's tip,
 * which runs on past the tip back among the last chevrons for `back` pixels:
 * where its birds shrink from the chevrons' length to the flock's by `shrink`
 * and cool from the fire's color to cream by `cool`, the stretch over which the
 * wake hands over to the ring around the fire, and the stretch over which its
 * own birds thin out while the scattered ring fills in. The ring's half width
 * swells and narrows once every `lobe` pixels from `phase`, its birds `perPx`
 * to a pixel of route, and both the wake and the ring draw from their own
 * seeds.
 */
const WAKE = {
  back     : 34,
  cool     : 420,
  handover : { from: 280, to: 480 },
  ring     : { lobe: 613.6, perPx: 0.98, phase: 0.144 },
  seeds    : { ring: 71, wake: 83 },
  shrink   : 300,
  thin     : { from: 300, to: 480 }
} as const

/**
 * Builds the profile that is `join` before `span` and `ring` past it, each
 * reading eased across the span from the one to the other, so the line never
 * thins where the two meet.
 */
function handover(join: Profile, ring: Profile, span: { from: number, to: number }): Profile {
  const share = (s: number): number => common.smoothstep(span.from, span.to, s)
  const eased = (s: number, a: number, b: number): number => a + (b - a) * share(s)
  return {
    color : (s, edge, x, y) => common.mix(join.color(s, edge, x, y), ring.color(s, edge, x, y), share(s)),
    perPx : s => eased(s, join.perPx(s), ring.perPx(s)),
    size  : (s, draw) => eased(s, join.size(s, draw), ring.size(s, draw)),
    width : s => eased(s, join.width(s), ring.width(s))
  }
}

/**
 * Reports whether the point at `x` and `y` falls inside `box` or within
 * `margin` pixels of it.
 */
const inside = (box: common.Box, x: number, y: number, margin: number): boolean =>
  x > box.x0 - margin && x < box.x1 + margin && y > box.y0 - margin && y < box.y1 + margin

/**
 * Builds the wake's own profile as it leaves the wordmark: narrow as the
 * chevrons' column at the tip and widening steadily, its birds filling in over
 * the first 340 pixels, the chevrons' length and the fire's color at the tip
 * shrinking to the flock's length and cooling to cream.
 */
function leaving({ chevron, cream, fire, lift }: Wake, tuning: typeof WAKE): Profile {
  return {
    perPx : s => 0.3 + 0.38 * common.smoothstep(0, 340, s),
    width : s => 7 + 0.085 * Math.max(0, s),

    color: (s, edge, x, y) => {
      const cooled = common.smoothstep(0, tuning.cool, s)
      return common.mix(cream, lift, 0.2 + 0.8 * (1 - cooled) + 0.45 * fire.near(x, y) * cooled + 0.5 * edge)
    },

    size: (s, draw) => {
      const grown = common.smoothstep(0, tuning.shrink, s)
      return chevron * (0.94 + 0.12 * draw) * (1 - grown) + ringSize(draw) * grown
    }
  }
}

/**
 * Sizes a bird of the ring from a draw in [0, 1), most of them small and a few
 * up to 8.2 pixels long.
 */
const ringSize = (draw: number): number => 3.6 + 4.6 * draw ** 1.6

/**
 * Builds the profile of the ring around the fire: its half width swelling and
 * narrowing in lobes, its birds a fixed number to a pixel, and each bird
 * warming toward the fire's color as it nears the fire's middle or the edge of
 * the heat.
 */
function ringed({ cream, fire, lift }: Wake, tuning: typeof WAKE): Profile {
  const { lobe, perPx, phase } = tuning.ring
  return {
    color : (_, edge, x, y) => common.mix(cream, lift, 0.2 + 0.45 * fire.near(x, y) + 0.5 * edge),
    perPx : () => perPx,
    size  : (_, draw) => ringSize(draw),
    width : s => 3 + 36 * (0.55 + 0.75 * Math.sin(Math.PI * s / lobe + phase) ** 2)
  }
}

/**
 * Scatters the ring's birds along the route from where the handover starts to
 * the route's end, as many to a pixel as `profile` sets, kept off the card's
 * edges and out of every box the words take with a margin of 18 pixels.
 */
function scattered(wake: Wake, profile: Profile, tuning: typeof WAKE): string {
  const { height, width } = CANVAS
  const { length }        = wake.route
  const { from }          = tuning.handover
  const at                = (along: number): number => from + along * (length - from)
  let [peak, total]       = [0, 0]
  for (let s = from; s < length; s += 2) {
    peak   = Math.max(peak, profile.perPx(s))
    total += profile.perPx(s) * 2
  }
  return scene.murmuration({
    color   : (along, edge, x, y) => profile.color(at(along), edge, x, y),
    count   : Math.round(total),
    density : along => profile.perPx(at(along)) / peak,
    glyphs  : wake.glyphs,
    seed    : tuning.seeds.ring,
    size    : (along, draw) => profile.size(at(along), draw),
    spine   : along => wake.route.point(at(along) / length),
    width   : along => profile.width(at(along)),
    gap     : (x, y) => {
      const off = x < -10 || x > width + 10 || y < -10 || y > height + 10
      return off || wake.boxes.some(box => inside(box, x, y, 18)) ? 2 : wake.fire.gap(x, y)
    }
  })
}

/**
 * Lays the wake's own birds along the route a pixel at a time, as many to a
 * pixel as `profile` sets, thinning out as the ring fills in. Behind the tip
 * they fly between the last chevrons' rows, past it they spread across the
 * wake's width with its two edges favored, and each one inside a box the words
 * take with a margin of 6 pixels or inside the heat is left out.
 */
function trailing(wake: Wake, profile: Profile, tuning: typeof WAKE): string {
  const draw            = common.mulberry(tuning.seeds.wake)
  const birds: string[] = []
  const { from, to }    = tuning.thin
  for (let s = -tuning.back; s < to; s++) {
    if (draw() > profile.perPx(s) * (1 - common.smoothstep(from, to, s))) continue
    const spread           = draw() * 2 - 1
    const across           = Math.sign(spread) * Math.abs(spread) ** (s < 30 ? 1 : 0.62) * profile.width(s)
    const [px, py, tx, ty] = wake.route.at(s + (draw() - 0.5) * 6)
    const lateral          = s < 0 ? (draw() < 0.5 ? -0.5 : 0.5) * wake.rows + (draw() - 0.5) * 2 : across
    const [x, y]           = [px - ty * lateral, py + tx * lateral]
    const heat             = wake.boxes.some(box => inside(box, x, y, 6)) ? 2 : wake.fire.gap(x, y)
    if (heat > 1) continue
    const edge   = Math.max(0, heat)
    const placed = Math.max(0, s)
    const angle  = Math.atan2(ty, tx) + (draw() - 0.5) * 0.4 * common.smoothstep(0, 200, s)
    const glyph  = Math.floor(draw() * wake.glyphs)
    const fill   = profile.color(placed, edge, x, y)
    birds.push(scene.parted({ angle, fill, glyph, size: profile.size(placed, draw()), x, y }, edge))
  }
  return birds.join('')
}

/**
 * Draws the flock leaving the wordmark as a widening wake: the line leaves the
 * tip as narrow as the chevrons' column, its first birds flying among the last
 * chevrons at their length and in the fire's color, and it widens and cools as
 * it runs on into the ring of birds around the fire.
 */
export function wakeOf(wake: Wake, tuning: typeof WAKE = WAKE): string {
  const { from, to } = tuning.thin
  const profile      = handover(leaving(wake, tuning), ringed(wake, tuning), tuning.handover)
  const filling      = (s: number): number => profile.perPx(s) * common.smoothstep(from, to, s)
  return trailing(wake, profile, tuning) + scattered(wake, { ...profile, perPx: filling }, tuning)
}
