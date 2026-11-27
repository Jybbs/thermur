import { TINY } from '../front/slots.ts'

/**
 * The measures the composite reports for a flock, as the package's `Flock` and
 * `Flight` compute them: its polarization, the length of the mean heading from
 * zero for headings that cancel to one for a flock flying as one, and the
 * groups it splits into, two drones sharing a group where a chain of hops no
 * longer than `reach` joins them, three of the plan's spacings in the
 * package. Each drone's east, north, and height sit at three places of the
 * positions and velocities a measure reads, and every array a measure writes
 * is built once for the flock's size.
 */
export class Measures {
  readonly #byEast  : (a: number, b: number) => number
  #east             : ArrayLike<number> = []
  readonly #order   : Int32Array
  readonly #parents : Int32Array
  readonly #reach   : number

  constructor(count: number, reach: number) {
    this.#order   = new Int32Array(count)
    this.#parents = new Int32Array(count)
    this.#reach   = reach
    this.#byEast  = (a, b) => this.#east[a * 3]! - this.#east[b * 3]! || a - b
  }

  /**
   * Finds the drone heading the group `drone` sits in, halving the path to it
   * on the way.
   */
  #root(drone: number): number {
    const parents = this.#parents
    let at        = drone
    while (parents[at] !== at) {
      parents[at] = parents[parents[at]!]!
      at          = parents[at]!
    }
    return at
  }

  /**
   * Counts the groups the drones at `positions` split into, sweeping the drones
   * in order east and joining each to every one after it no farther than the
   * reach, so only drones within the reach east of each other are compared.
   */
  groups(positions: ArrayLike<number>): number {
    const order   = this.#order
    const parents = this.#parents
    const reach   = this.#reach
    let groups    = order.length
    for (let drone = 0; drone < order.length; drone++) {
      order[drone]   = drone
      parents[drone] = drone
    }
    this.#east = positions
    order.sort(this.#byEast)
    for (let first = 0; first < order.length; first++) {
      const a = order[first]! * 3
      for (let next = first + 1; next < order.length; next++) {
        const b = order[next]! * 3
        if (positions[b]! - positions[a]! > reach) break
        const apart = Math.hypot(
          positions[b]! - positions[a]!,
          positions[b + 1]! - positions[a + 1]!,
          positions[b + 2]! - positions[a + 2]!
        )
        if (apart > reach) continue
        const one = this.#root(a / 3), other = this.#root(b / 3)
        if (one === other) continue
        parents[Math.max(one, other)] = Math.min(one, other)
        groups--
      }
    }
    return groups
  }

  /**
   * Measures the polarization of the drones flying at `velocities`.
   */
  polarization(velocities: ArrayLike<number>): number {
    let x = 0, y = 0, z = 0
    for (let at = 0; at < velocities.length; at += 3) {
      const speed = Math.max(Math.hypot(velocities[at]!, velocities[at + 1]!, velocities[at + 2]!), TINY)
      x          += velocities[at]! / speed
      y          += velocities[at + 1]! / speed
      z          += velocities[at + 2]! / speed
    }
    return Math.hypot(x, y, z) / Math.max(velocities.length / 3, 1)
  }
}
