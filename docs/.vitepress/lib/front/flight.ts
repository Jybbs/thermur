import { type Geometry, NARROW } from '../canvas/part.ts'
import * as common               from '../common.ts'
import { type Heat, repel }      from './heat.ts'
import { Ring }                  from './ring.ts'

/**
 * The eight's flight as `flight` flies it: from `start`, after a roost at
 * `roostX` and `roostY` for each moment, sampled `samples` times twenty times a
 * second through `fire` where one is given, gliding onto `perches` past the
 * landing time `land`, inside a page `w` pixels wide.
 */
interface Course {
  land    : number
  perches : readonly (readonly [number, number])[]
  roostX  : (t: number) => number
  roostY  : (t: number) => number
  samples : number
  start   : readonly [number, number]
  w       : number
  fire?   : Heat
}

/**
 * The eight birds as they fly: where each is, how fast it moves, and the force
 * on it this step.
 */
interface Eight {
  ax : Float32Array
  ay : Float32Array
  vx : Float32Array
  vy : Float32Array
  x  : Float32Array
  y  : Float32Array
}

/**
 * The eight's paths, each bird's samples smoothed, and the heat it felt at
 * each.
 */
type Paths = Pick<Run, 'hs' | 'xs' | 'ys'>

/**
 * The eight's flight down the page: each bird's path through the samples,
 * smoothed, and the heat it felt at each, beside the fire they part around, the
 * height of the ridge at the page's foot they land on and the height its waves
 * cancel at, `base`, and where the flight starts.
 */
export interface Run {
  base  : number
  fire  : Ring
  hs    : readonly Float32Array[]
  ridge : (x: number) => number
  xs    : readonly Float32Array[]
  y0    : number
  ys    : readonly Float32Array[]
}

/**
 * The number of birds the lines follow, one drone and its seven nearest
 * neighbors.
 */
export const EIGHT = 8
/**
 * The waves of the ridge at the page's foot the eight land on.
 */
const FOOT = { amplitude: 12, periods: [130, 53, 21], phases: [0.4, 1.6, 2.8] } as const
/**
 * The times a second the flight keeps where each bird was.
 */
const HZ = 20
/**
 * The times `smoothPath` cuts the corners of a path, each pass doubling its
 * samples.
 */
const PASSES = 2
/**
 * The steps the flight takes between two samples it keeps.
 */
const SUBSTEPS = 3
/**
 * The samples `smoothPath` writes for each one the flight keeps.
 */
const scale = 2 ** PASSES
/**
 * The length of each step the flight takes, in seconds.
 */
const step = 1 / (HZ * SUBSTEPS)
/**
 * The samples a run holds for each second of the flight, once each sample the
 * flight keeps is smoothed into `scale` of them.
 */
export const samplesPerSecond = HZ * scale

/**
 * Builds one value for each of the eight birds.
 */
const birds = (): Float32Array => new Float32Array(EIGHT)

/**
 * Flies the eight after a roost from one seed, keeping where each was and the
 * heat it felt twenty times a second, then smooths each bird's path.
 */
function flight(course: Course): Paths {
  const random = common.mulberry(97)
  const eight  = { ax: birds(), ay: birds(), vx: birds(), vy: birds(), x: birds(), y: birds() }
  for (let i = 0; i < EIGHT; i++) {
    eight.x[i]  = course.start[0] + (random() - 0.5) * 16
    eight.y[i]  = course.start[1] + (random() - 0.5) * 8
    eight.vx[i] = -40 - random() * 20
    eight.vy[i] = 60 + random() * 20
  }
  const { samples }  = course
  const rows         = (): Float32Array[] => Array.from({ length: EIGHT }, () => new Float32Array(samples))
  const [xs, ys, hs] = [rows(), rows(), rows()]
  const out: [number, number] = [0, 0]
  for (let k = 0, t = 0; k < samples; k++) {
    for (let i = 0; i < EIGHT; i++) {
      xs[i]![k] = eight.x[i]!
      ys[i]![k] = eight.y[i]!
      hs[i]![k] = course.fire?.heat(eight.x[i]!, eight.y[i]!) ?? 0
    }
    for (let sub = 0; sub < SUBSTEPS; sub++, t += step) {
      const landing = common.clamp((t - course.land) / 2.5)
      steer(eight, course, landing, out, random, t)
      move(eight, landing)
    }
  }
  const widened = (row: Float32Array): Float32Array =>
    Float32Array.from({ length: samples * scale }, (_, k) => row[Math.floor(k / scale)]!)
  return {
    hs : hs.map(row => widened(row)),
    xs : xs.map(row => smoothPath(row)),
    ys : ys.map(row => smoothPath(row))
  }
}

/**
 * Moves the eight on by one step along the forces set on them, holding each
 * bird's speed between its floor and 125 pixels a second, the floor falling
 * away as the landing, from 0 to 1, runs its course.
 */
function move(eight: Eight, landing: number): void {
  const { ax, ay, vx, vy, x, y } = eight
  const floor                    = 70 * (1 - landing)
  for (let i = 0; i < EIGHT; i++) {
    vx[i]       = vx[i]! + ax[i]! * step
    vy[i]       = vy[i]! + ay[i]! * step
    const speed = Math.hypot(vx[i]!, vy[i]!) + 1e-6
    const want  = common.clamp(speed, floor, 125)
    vx[i]       = vx[i]! * want / speed
    vy[i]       = vy[i]! * want / speed
    x[i]        = x[i]! + vx[i]! * step
    y[i]        = y[i]! + vy[i]! * step
  }
}

/**
 * Sets the force on every bird for one step at `t`, writing the push away from
 * the heat into `out`: toward the others' heading and middle, held less inside
 * the heat, toward the roost, apart from any bird nearer than 20 pixels, down
 * the heat's slope, back from the page's edges, and onto its perch as `landing`
 * runs from 0 to 1.
 */
function steer(
  eight   : Eight,
  course  : Course,
  landing : number,
  out     : [number, number],
  random  : () => number,
  t       : number
): void {
  const { ax, ay, vx, vy, x, y }             = eight
  const { fire, perches, roostX, roostY, w } = course
  let mvx                                    = 0, mvy = 0, mx = 0, my = 0
  for (let i = 0; i < EIGHT; i++) {
    mx  += x[i]!
    my  += y[i]!
    mvx += vx[i]!
    mvy += vy[i]!
  }
  for (let i = 0; i < EIGHT; i++) {
    const hot  = fire?.heat(x[i]!, y[i]!) ?? 0
    const hold = 1 - 0.85 * Math.min(1, hot * 5)
    const ox   = (mx - x[i]!) / (EIGHT - 1)
    const oy   = (my - y[i]!) / (EIGHT - 1)
    let fx     = (((mvx - vx[i]!) / (EIGHT - 1) - vx[i]!) * 1.8 + (ox - x[i]!) * 0.6) * hold
               + (roostX(t) - x[i]!) * 0.8 * hold + (random() - 0.5) * 14
    let fy = (((mvy - vy[i]!) / (EIGHT - 1) - vy[i]!) * 1.8 + (oy - y[i]!) * 0.6) * hold
           + (roostY(t) - y[i]!) * 0.8 + (random() - 0.5) * 14
    for (let j = 0; j < EIGHT; j++) {
      const dx = x[i]! - x[j]!
      const dy = y[i]! - y[j]!
      const d  = Math.hypot(dx, dy) + 0.01
      fx      += j !== i && d < 20 ? dx / d * (20 - d) * 6 : 0
      fy      += j !== i && d < 20 ? dy / d * (20 - d) * 6 : 0
    }
    if (fire !== undefined && hot > 0.004) {
      repel(fire, x[i]!, y[i]!, hot, 1500, out)
      fx += out[0]
      fy += out[1] * 0.35
    }
    fx            += Math.max(0, 18 - x[i]!) * 14 - Math.max(0, x[i]! - (w - 18)) * 14
    const [px, py] = perches[i]!
    ax[i]          = fx * (1 - landing) + ((px - x[i]!) * 6 - vx[i]! * 4.5) * landing
    ay[i]          = fy * (1 - landing) + ((py - y[i]!) * 6 - vy[i]! * 4.5) * landing
  }
}

/**
 * Finds where on the page the pen follows the reader to with the page scrolled
 * `dy` pixels, low in the window at the top of the page and lower still at its
 * foot.
 */
export const penAt = (g: Geometry, dy: number): number =>
  dy + g.h * (0.62 + 0.34 * common.clamp(dy / Math.max(1, g.page - g.h)))

/**
 * Plans the eight's flight down the page from behind the hero's ridge: a roost
 * sweeping side to side as it sinks, holding its line where the fire waits so
 * the eight come straight down onto it, and slowing to the ridge at the page's
 * foot. A dry flight with no fire finds where the eight pass the turn, and the
 * fire is set there before the flight is flown again.
 */
export function plan(g: Geometry, horizon: number, fire: number): Run {
  const y0    = horizon + 34
  const start = [g.w * (g.w < NARROW ? 0.84 : 0.88), y0] as const
  const v     = 50
  const amp   = g.w * (g.w < NARROW ? 0.28 : 0.32)
  const phase = Math.asin(common.clamp((start[0] - g.w * 0.5) / amp, -1, 1))
  const sweep = (t: number): number => {
    const x = g.w * (0.5 + 0.05 * Math.sin(t * 0.47)) + amp * Math.sin(phase + t * 0.2)
    return common.clamp(x, 20, g.w - 20)
  }
  const foot    = g.page - 120
  const land    = (foot - y0) / v
  const middle  = penAt(g, 0.5 * Math.max(0, g.page - g.h))
  const turn    = (common.clamp(middle, y0 + 120, foot - 160) - y0) / v
  const held    = sweep(turn)
  const base    = g.page - 70
  const ridge   = common.ridgeline({ ...FOOT, base })
  const perches = Array.from({ length: EIGHT }, (_, i): readonly [number, number] => {
    const px = g.w * (0.58 + i * 0.035)
    return [px, ridge(px) - 3]
  })
  const course = {
    land    : land,
    perches : perches,
    roostY  : (t: number): number => Math.min(y0 + v * t, foot),
    samples : Math.max(8, Math.ceil((land + 4.5) * HZ) + 1),
    start   : start,
    w       : g.w,
    roostX  : (t: number): number => {
      const k = Math.exp(-(((t - turn) / 2.6) ** 2))
      return sweep(t) * (1 - k) + held * k
    }
  }
  const dry  = flight(course)
  const k    = Math.min(dry.xs[0]!.length - 1, Math.round((turn + 0.8) * samplesPerSecond))
  const ring = new Ring({
    fire   : fire,
    radius : 86,
    x      : dry.xs.reduce((sum, row) => sum + row[k]!, 0) / EIGHT,
    y      : dry.ys.reduce((sum, row) => sum + row[k]!, 0) / EIGHT
  })
  return { ...flight({ ...course, fire: ring }), base, fire: ring, ridge, y0 }
}

/**
 * Smooths one axis of a path by cutting its corners twice, so a line drawn
 * through it bends without kinks, each pass doubling its samples.
 */
export function smoothPath(values: ArrayLike<number>): Float32Array {
  let points = Array.from(values)
  for (let pass = 0; pass < PASSES; pass++) {
    const cut = [points[0]!]
    for (let i = 0; i < points.length - 1; i++) {
      cut.push(0.75 * points[i]! + 0.25 * points[i + 1]!, 0.25 * points[i]! + 0.75 * points[i + 1]!)
    }
    cut.push(points.at(-1)!)
    points = cut
  }
  return Float32Array.from(points)
}
