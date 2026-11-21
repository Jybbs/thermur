/**
 * The order of a flock's drones along the Z-order curve through their
 * positions, each axis cut into 1,024 steps across the drones' extent, as the
 * package's `zorder` sets it, so drones near one another mostly sit near one
 * another in the order. Every array it writes is built once.
 */
export class Curve {
  /**
   * The drones in the order of their codes along the curve.
   */
  readonly order   : Int32Array
  /**
   * Each drone's place in `order`.
   */
  readonly places  : Int32Array
  /**
   * Orders two drones by their codes, a tie going to the lower index.
   */
  readonly #byCode : (a: number, b: number) => number
  readonly #codes  : Int32Array

  constructor(count: number) {
    this.order   = new Int32Array(count)
    this.places  = new Int32Array(count)
    this.#codes  = new Int32Array(count)
    this.#byCode = (a, b) => this.#codes[a]! - this.#codes[b]! || a - b
  }

  /**
   * Finds the drone `offset` places from the drone at `index` along the curve,
   * clipped to the curve's ends.
   */
  beside(index: number, offset: number): number {
    return this.order[Math.min(Math.max(this.places[index]! + offset, 0), this.order.length - 1)]!
  }

  /**
   * Orders the drones by their positions in `points`, the first at `base` and
   * each next `stride` past the one before.
   */
  through(base: number, points: Float64Array, stride: number): void {
    const count = this.order.length
    this.#codes.fill(0)
    for (let axis = 0; axis < 3; axis++) {
      let high = -Infinity, low = Infinity
      for (let index = 0; index < count; index++) {
        const at = points[base + index * stride + axis]!
        high     = Math.max(high, at)
        low      = Math.min(low, at)
      }
      const span = high - low > 0 ? high - low : 1
      for (let index = 0; index < count; index++) {
        const steps = Math.trunc((points[base + index * stride + axis]! - low) * 1023 / span)
        let code    = this.#codes[index]!
        for (let bit = 0; bit < 10; bit++) code |= ((steps >> bit) & 1) << (3 * bit + axis)
        this.#codes[index] = code
      }
    }
    for (let index = 0; index < count; index++) this.order[index] = index
    this.order.sort(this.#byCode)
    for (let place = 0; place < count; place++) this.places[this.order[place]!] = place
  }
}
