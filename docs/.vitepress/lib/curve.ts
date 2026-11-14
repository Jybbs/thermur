import { type Pair, clamp } from './common.ts'
import { tenths }           from './scene.ts'

/**
 * A point on a curve beside the heading there, as the point's x and y and the x
 * and y of a unit vector pointing along the curve.
 */
type Heading = readonly [number, number, number, number]

/**
 * The points a curve is sampled at between each pair of the points it runs
 * through.
 */
const SAMPLES = 24

/**
 * Reads the four points the spline's span from `points[index]` to the point
 * after it is drawn from, the first and the last point standing in for the
 * missing neighbor at either end.
 */
function around(points: readonly Pair[], index: number): [Pair, Pair, Pair, Pair] {
  const at = (place: number): Pair => points[clamp(place, 0, points.length - 1)]!
  return [at(index - 1), at(index), at(index + 1), at(index + 2)]
}

/**
 * Finds one coordinate of the Catmull-Rom spline from `p1` to `p2` at `t`, `p0`
 * and `p3` being the points before and after them.
 */
function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  return 0.5 * (
      2 * p1
    + (p2 - p0) * t
    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
    + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3
  )
}

/**
 * Samples the Catmull-Rom spline through `points`, `SAMPLES` points between
 * each pair of them.
 */
function sampled(points: readonly Pair[]): Pair[] {
  const dense: Pair[] = []
  for (let index = 0; index < points.length - 1; index++) {
    const [p0, p1, p2, p3] = around(points, index)
    for (let step = 0; step < SAMPLES; step++) {
      const t = step / SAMPLES
      dense.push([catmullRom(p0[0], p1[0], p2[0], p3[0], t), catmullRom(p0[1], p1[1], p2[1], p3[1], t)])
    }
  }
  dense.push(points.at(-1)!)
  return dense
}

/**
 * Writes a point as an SVG path writes it, each coordinate rounded to a tenth
 * of a pixel.
 */
const written = ([x, y]: Pair): string => `${tenths(x)},${tenths(y)}`

/**
 * A smooth curve through a list of points, sampled along its length so a
 * distance along it reads evenly however the points are spaced.
 */
export class Curve {
  readonly length   : number
  readonly #lengths : readonly number[]
  readonly #points  : readonly Pair[]

  constructor(through: readonly Pair[]) {
    const points  = sampled(through)
    const lengths = [0]
    for (let index = 1; index < points.length; index++) {
      const [[x0, y0], [x1, y1]] = [points[index - 1]!, points[index]!]
      lengths.push(lengths[index - 1]! + Math.hypot(x1 - x0, y1 - y0))
    }
    this.#points  = points
    this.#lengths = lengths
    this.length   = lengths.at(-1)!
  }

  /**
   * Finds the point `s` pixels along the curve and the heading there, measured
   * across the six pixels around it. For `s` below zero, the point runs on `s`
   * pixels back along the heading the curve starts on.
   */
  at(s: number): Heading {
    const clamped  = Math.max(0, s)
    const [ax, ay] = this.point((clamped - 3) / this.length)
    const [bx, by] = this.point((clamped + 3) / this.length)
    const span     = Math.hypot(bx - ax, by - ay)
    const [tx, ty] = [(bx - ax) / span, (by - ay) / span]
    const [x, y]   = this.point(clamped / this.length)
    return s < 0 ? [x + tx * s, y + ty * s, tx, ty] : [x, y, tx, ty]
  }

  /**
   * Finds the point `u` of the way along the curve, `u` held between 0 and 1.
   */
  point(u: number): Pair {
    const lengths   = this.#lengths
    const target    = clamp(u) * this.length
    let [low, high] = [0, lengths.length - 1]
    while (high - low > 1) {
      const middle = (low + high) >> 1
      if (lengths[middle]! < target) low = middle
      else high = middle
    }
    const [from, to] = [this.#points[low]!, this.#points[high]!]
    const t          = (target - lengths[low]!) / (lengths[high]! - lengths[low]!)
    return [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t]
  }
}

/**
 * Writes the SVG path of the spline `Curve` samples through `points`, as one
 * cubic Bézier for each span between two of them.
 */
export function pathThrough(points: readonly Pair[]): string {
  const spans = points.slice(1).map((_, index) => {
    const [p0, p1, p2, p3] = around(points, index)
    const first: Pair      = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6]
    const second: Pair     = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6]
    return `C${[first, second, p2].map(point => written(point)).join(' ')}`
  })
  return `M${written(points[0]!)}${spans.join('')}`
}
