import source from '../rendered/credit.json' with { type: 'json' }

/**
 * The seal in the pixels of its viewbox, where:
 *
 * - `core`, `edge`, and `hair` are the radii of its three rings, the rim's
 *   words running between the core's and the hairline's.
 * - `flame` is the site's short fire on the ground at `y`, its crest `rise`
 *   pixels above, `half` pixels either side of the middle, under a glow reaching
 *   `glow` pixels.
 * - `rim` holds the radius each arc's baseline stands at, the top one's letters
 *   rising outward and the foot one's inward, and the radius `side` the dots of
 *   radius `dot` stand at between them.
 */
export const SEAL = {
  center : 60,
  core   : 32,
  edge   : 58,
  flame  : { glow: 29, half: 15, rise: 4, y: 72 },
  hair   : 54.5,
  rim    : { dot: 1.1, foot: 46.6, side: 43.25, top: 39.9 },
  size   : 120
} as const

/**
 * The ground from the seal's middle to the core's ring at the height the flame
 * stands at, in the same pixels.
 */
export const GROUND = Math.sqrt(SEAL.core ** 2 - (SEAL.flame.y - SEAL.center) ** 2)

/**
 * Traces a circle of `radius` about the seal's center as the path an arc of its
 * rim is set along, whose middle is the seal's head running clockwise from the
 * foot where `up` holds, and otherwise its foot running the other way from the
 * head.
 */
export function rimPath(radius: number, up: boolean): string {
  const { center }      = SEAL
  const [start, across] = up ? [center + radius, -2 * radius] : [center - radius, 2 * radius]
  const arc             = (down: number): string => `a${radius} ${radius} 0 0 ${up ? 1 : 0} 0 ${down}`
  return `M${center} ${start}${arc(across)}${arc(-across)}`
}

/**
 * Reads the words on each arc of the seal's rim out of a credit as
 * `docs:render` writes it: the creators' surnames along its top, and the year
 * and the license along its foot.
 */
export const rimOf = ({ license, surnames, year } = source): { foot: string, top: string } => ({
  foot : `${year} · ${license}`,
  top  : surnames.join(' & ')
})
