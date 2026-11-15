import source from '../rendered/composite.json' with { type: 'json' }

/**
 * The default slots and the fields of the plan `mise render` writes out of the
 * package's settings, each slot under its kind.
 */
export type Model = typeof source

/**
 * The kind of component each slot holds that `Composite` copies, which a model
 * naming any other fails to build.
 */
const KINDS = { cohesion: 'Morse', contact: 'SoftCore', coupling: 'Constant', heading: 'Overdamped', noise: 'White' }
/**
 * The smallest positive single-precision value, which a length divides by in
 * place of zero, as the package's `unit` takes it.
 */
const TINY = 1.1754944e-38

/**
 * Writes the indices of the drone at `index`'s `best.length` nearest other
 * drones into `into` from `base` on, nearest first, the drones laid out as
 * east, north, and height in `points`, and `best` holding the squared distances
 * found so far.
 */
export function ranked(
  best   : Float64Array,
  index  : number,
  into   : Int32Array,
  points : Float64Array,
  base   : number = 0
): void {
  const last = best.length - 1
  const at   = index * 3
  best.fill(Infinity)
  for (let o = 0, other = 0; o < points.length; o += 3, other++) {
    const distance = (points[o]! - points[at]!) ** 2
                   + (points[o + 1]! - points[at + 1]!) ** 2
                   + (points[o + 2]! - points[at + 2]!) ** 2
    if (other === index || distance >= best[last]!) continue
    let k = last
    for (; k > 0 && best[k - 1]! > distance; k--) {
      best[k]        = best[k - 1]!
      into[base + k] = into[base + k - 1]!
    }
    best[k]        = distance
    into[base + k] = other
  }
}

/**
 * The package's composite under its default slots, stepped as one Euler step of
 * the plan's `step_s` the way a flight steps it: each drone senses its nearest
 * neighbors, sums the Morse cohesion, the soft-core contact, and the alignment
 * at the constant coupling, keeps the part of that sum across its velocity, and
 * moves on its old velocity while the sum and a Brownian increment scaled by the
 * white noise's amplitude move its velocity, which then returns to the cruise
 * speed. Each drone's east, north, and height sit in meters at three places of
 * `positions`, and its velocity in meters per second likewise in `velocities`.
 * Every array a step reads and writes is built once, so a step allocates
 * nothing.
 */
export class Composite {
  readonly count          : number
  readonly model          : Model
  readonly positions      : Float64Array
  readonly velocities     : Float64Array
  readonly #accelerations : Float64Array
  readonly #best          : Float64Array
  readonly #near          : Int32Array

  constructor(count: number, model: Model = source) {
    for (const [slot, kind] of Object.entries(KINDS)) {
      const held = model[slot as keyof typeof KINDS].kind
      if (held !== kind) throw new Error(`The ${slot} slot holds ${held}, where the copy flies ${kind}`)
    }
    this.count          = count
    this.model          = model
    this.positions      = new Float64Array(count * 3)
    this.velocities     = new Float64Array(count * 3)
    this.#accelerations = new Float64Array(count * 3)
    this.#best          = new Float64Array(model.plan.neighbors)
    this.#near          = new Int32Array(count * model.plan.neighbors)
  }

  /**
   * Sums the slots for the drone at `index` and the push `pushes` holds for it,
   * keeping the part across its velocity, into `#accelerations`.
   */
  #accelerate(index: number, pushes: ArrayLike<number>): void {
    const { cohesion, contact, coupling, plan } = this.model
    const { positions, velocities }             = this
    const at                                    = index * 3
    let ax = pushes[at]!, ay = pushes[at + 1]!, az = pushes[at + 2]!
    for (let k = 0; k < plan.neighbors; k++) {
      const o        = this.#near[index * plan.neighbors + k]! * 3
      const dx       = positions[o]! - positions[at]!
      const dy       = positions[o + 1]! - positions[at + 1]!
      const dz       = positions[o + 2]! - positions[at + 2]!
      const distance = Math.hypot(dx, dy, dz)
      const pull     = cohesion.attraction_m2_s2 / cohesion.attraction_m * Math.exp(-distance / cohesion.attraction_m)
                     - cohesion.repulsion_m2_s2 / cohesion.repulsion_m * Math.exp(-distance / cohesion.repulsion_m)
                     - contact.stiffness_m_s2 * Math.max(1 - distance / contact.diameter_m, 0)
      const along    = pull / Math.max(distance, TINY)
      ax += along * dx + coupling.coupling_per_s * (velocities[o]! - velocities[at]!)
      ay += along * dy + coupling.coupling_per_s * (velocities[o + 1]! - velocities[at + 1]!)
      az += along * dz + coupling.coupling_per_s * (velocities[o + 2]! - velocities[at + 2]!)
    }
    const vx    = velocities[at]!, vy = velocities[at + 1]!, vz = velocities[at + 2]!
    const speed = Math.max(Math.hypot(vx, vy, vz), TINY)
    const share = (ax * vx + ay * vy + az * vz) / (speed * speed)
    this.#accelerations[at]     = ax - share * vx
    this.#accelerations[at + 1] = ay - share * vy
    this.#accelerations[at + 2] = az - share * vz
  }

  /**
   * Moves every drone through one step under the Brownian `increments` and the
   * accelerations `pushes` adds to the slots, each laid out as `positions` is.
   */
  advance(increments: ArrayLike<number>, pushes: ArrayLike<number>): void {
    const { cruise_m_s, neighbors, step_s } = this.model.plan
    const amplitude                         = this.model.noise.amplitude
    const { positions, velocities }         = this
    for (let index = 0; index < this.count; index++) ranked(this.#best, index, this.#near, positions, index * neighbors)
    for (let index = 0; index < this.count; index++) this.#accelerate(index, pushes)
    for (let at = 0; at < positions.length; at += 3) {
      for (let axis = at; axis < at + 3; axis++) {
        positions[axis]  = positions[axis]! + step_s * velocities[axis]!
        velocities[axis] = velocities[axis]! + step_s * this.#accelerations[axis]! + amplitude * increments[axis]!
      }
      const scale = cruise_m_s / Math.max(Math.hypot(velocities[at]!, velocities[at + 1]!, velocities[at + 2]!), TINY)
      for (let axis = at; axis < at + 3; axis++) velocities[axis] = velocities[axis]! * scale
    }
  }

  /**
   * Launches the drones around the origin as the plan launches a flock, in a
   * Gaussian cloud whose length, width, and depth the plan's axes set along the
   * course, across it, and up, holding the plan's spacing cubed for each drone,
   * each flying the course within its scatter at the cruise speed, every draw
   * from `normal`.
   */
  launch(normal: () => number): void {
    const { axes, course_deg, cruise_m_s, scatter_deg, spacing_m } = this.model.plan
    const course  = course_deg * Math.PI / 180
    const scatter = scatter_deg * Math.PI / 180
    const scale   = spacing_m * Math.cbrt(this.count * 3 / (4 * Math.PI)) / Math.cbrt(axes[0]! * axes[1]! * axes[2]!)
    const basis   = [Math.cos(course), Math.sin(course), -Math.sin(course), Math.cos(course)]
    for (let at = 0; at < this.positions.length; at += 3) {
      const along           = normal() * axes[0]! * scale
      const across          = normal() * axes[1]! * scale
      this.positions[at]     = along * basis[0]! + across * basis[2]!
      this.positions[at + 1] = along * basis[1]! + across * basis[3]!
      this.positions[at + 2] = normal() * axes[2]! * scale
    }
    for (let at = 0; at < this.velocities.length; at += 3) {
      const x     = basis[0]! + normal() * scatter
      const y     = basis[1]! + normal() * scatter
      const z     = normal() * scatter
      const speed = cruise_m_s / Math.hypot(x, y, z)
      this.velocities[at]     = x * speed
      this.velocities[at + 1] = y * speed
      this.velocities[at + 2] = z * speed
    }
  }
}
