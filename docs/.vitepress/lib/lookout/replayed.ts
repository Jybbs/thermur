import type { Playing, Watched } from './flight.ts'

/**
 * A flight `thermur watch` served, played between its frames by straight
 * lines and held on its last frame once it ends, its first frame kept the
 * seconds between two frames after the launch.
 */
export class Replayed implements Playing {
  readonly count      : number
  readonly positions  : Float64Array
  readonly velocities : Float64Array
  #seconds            = 0
  readonly #watched   : Watched

  constructor(watched: Watched) {
    this.count      = watched.drones
    this.positions  = new Float64Array(watched.drones * 3)
    this.velocities = new Float64Array(watched.drones * 3)
    this.#watched   = watched
    this.play(0)
  }

  get seconds(): number {
    return this.#seconds + this.#watched.frame_s
  }

  play(dt: number): void {
    const { frame_s, positions, velocities } = this.#watched
    const block  = this.positions.length
    const last   = positions.length / block - 1
    this.#seconds = Math.min(this.#seconds + dt, last * frame_s)
    const frame  = Math.min(Math.floor(this.#seconds / frame_s), Math.max(last - 1, 0))
    const share  = Math.min(this.#seconds / frame_s - frame, 1)
    const from   = frame * block
    const to     = Math.min(frame + 1, last) * block
    for (let k = 0; k < block; k++) {
      this.positions[k]  = positions[from + k]! + share * (positions[to + k]! - positions[from + k]!)
      this.velocities[k] = velocities[from + k]! + share * (velocities[to + k]! - velocities[from + k]!)
    }
  }
}
