import type { Geometry } from '../canvas/part.ts'
import type { Front }    from './front.ts'

/**
 * A mark on one of the plot's axes, at its value and as it is written.
 */
interface Tick {
  value : number
  word  : string
}

/**
 * The distance east the plot runs to, in meters, a round reach showing the
 * front and the air it has yet to reach.
 */
const ACROSS = 300
/**
 * The height beneath the plot, in pixels, holding the distances, the axis's
 * name, and the rail.
 */
const FOOT = 78
/**
 * The space left of the plot for the temperatures, in pixels.
 */
const LEFT = 34
/**
 * The forms of the distance axis's name, the longest first.
 */
const NAMES = ['Distance east of the western edge (m)', 'Distance east (m)']
/**
 * How far up from the canvas's foot the rail sits, in pixels.
 */
const RAIL = 15
/**
 * The space right of the plot, in pixels, holding the last cell's mark.
 */
const RIGHT = 10
/**
 * The pixels between the points the air's curve is drawn through.
 */
const SAMPLE = 3
/**
 * The temperatures the plot marks are the multiples of this, in kelvin.
 */
const TICK_K = 10
/**
 * The space above the plot for the temperature axis's name, in pixels.
 */
const TOP = 26

/**
 * The plot's axes as they stand at one fit: its box, the pixel of a distance
 * and of a temperature, the longest name of the distance axis that fits, the
 * arrays the air's curve is written into, and the paths that stand still until
 * the next fit, meaning the rules, the trace of each snapshot, the area of the
 * last one, and the rail's track and ticks.
 */
export class Axes {
  readonly columns : number
  readonly front   : Front
  readonly height  : string
  readonly high    : number
  readonly kelvins : readonly Tick[]
  readonly left    = LEFT
  readonly low     : number
  readonly meters  : readonly Tick[] = [0, 100, 200, 300].map(value => ({ value, word: String(value) }))
  readonly right   = RIGHT
  finale           : Path2D | undefined
  grid             : Path2D | undefined
  name             = NAMES[0]!
  railY            = 0
  reaches          = new Float32Array()
  traces           : readonly Path2D[] = []
  track            : Path2D | undefined
  x0               = LEFT
  x1               = 0
  xs               = new Float32Array()
  y0               = TOP
  y1               = 0
  ys               = new Float32Array()

  constructor(front: Front) {
    this.front       = front
    this.columns     = Math.floor(ACROSS / front.spacing + 0.5)
    this.height      = `Air temperature at ${front.lowest} m (K)`
    this.low         = front.air(front.lowest, 0, Number.POSITIVE_INFINITY) - 4
    this.high        = front.air(front.lowest, 0, Number.NEGATIVE_INFINITY) + 2
    const [from, to] = [Math.ceil(this.low / TICK_K), Math.floor(this.high / TICK_K)]
    this.kelvins     = Array.from({ length: to - from + 1 }, (_, index) => ({
      value : (from + index) * TICK_K,
      word  : String((from + index) * TICK_K)
    }))
  }

  /**
   * Builds the rules: one at each temperature marked, the distance axis, and a
   * tick at each distance marked.
   */
  #rules(): Path2D {
    const path = new Path2D()
    for (const { value } of this.kelvins) {
      path.moveTo(this.x0, this.down(value))
      path.lineTo(this.x1, this.down(value))
    }
    for (const { value } of this.meters) {
      path.moveTo(this.across(value), this.y1)
      path.lineTo(this.across(value), this.y1 + 4)
    }
    path.moveTo(this.x0, this.y1)
    path.lineTo(this.x1, this.y1)
    return path
  }

  /**
   * Builds the path of the air's curve after `seconds`.
   */
  #trace(seconds: number): Path2D {
    const path = new Path2D()
    for (let index = 0; index < this.xs.length; index++) {
      const y = this.down(this.front.air(this.front.lowest, seconds, this.reaches[index]!))
      if (index === 0) path.moveTo(this.xs[index]!, y)
      else path.lineTo(this.xs[index]!, y)
    }
    return path
  }

  /**
   * Builds the rail's track and a tick at each snapshot.
   */
  #rail(): Path2D {
    const { snapshots } = this.front
    const path          = new Path2D()
    path.moveTo(this.x0, this.railY)
    path.lineTo(this.x1, this.railY)
    for (let index = 0; index < snapshots; index++) {
      const x = this.x0 + index / (snapshots - 1) * (this.x1 - this.x0)
      path.moveTo(x, this.railY - 4)
      path.lineTo(x, this.railY + 4)
    }
    return path
  }

  /**
   * Reads the pixel across the plot `meters` east of the western edge.
   */
  across(meters: number): number {
    return this.x0 + meters / ACROSS * (this.x1 - this.x0)
  }

  /**
   * Reads the pixel down the plot of a temperature in kelvin.
   */
  down(kelvin: number): number {
    return this.y1 - (kelvin - this.low) / (this.high - this.low) * (this.y1 - this.y0)
  }

  /**
   * Fits the axes to the canvas, measuring each name of the distance axis
   * through `pen` and building the arrays and the paths.
   */
  fit(g: Geometry, pen: CanvasRenderingContext2D): void {
    const { interval, snapshots } = this.front
    this.x1    = g.w - RIGHT
    this.y1    = g.h - FOOT
    this.railY = g.h - RAIL
    this.name  = NAMES.find(name => {
      const { actualBoundingBoxLeft: left, actualBoundingBoxRight: right } = pen.measureText(name)
      return left + right <= this.x1 - this.x0
    }) ?? NAMES.at(-1)!
    const points = Math.ceil((this.x1 - this.x0) / SAMPLE) + 1
    this.reaches = Float32Array.from({ length: points }, (_, index) => index / (points - 1) * ACROSS)
    this.xs      = this.reaches.map(meters => this.across(meters))
    this.ys      = new Float32Array(points)
    this.grid    = this.#rules()
    this.traces  = Array.from({ length: snapshots }, (_, index) => this.#trace(index * interval))
    this.finale  = this.#trace((snapshots - 1) * interval)
    this.finale.lineTo(this.x1, this.y1)
    this.finale.lineTo(this.x0, this.y1)
    this.finale.closePath()
    this.track   = this.#rail()
  }

  /**
   * Writes the air's curve after `seconds` into the array it is drawn from.
   */
  write(seconds: number): void {
    for (let index = 0; index < this.ys.length; index++) {
      this.ys[index] = this.down(this.front.air(this.front.lowest, seconds, this.reaches[index]!))
    }
  }
}
