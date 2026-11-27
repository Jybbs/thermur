import { FLIGHT } from '../front/flock.ts'
import { Front }  from '../threat/front.ts'
import threat     from '../rendered/threat.json' with { type: 'json' }

/**
 * A fire lit at the middle of the roost at the flight's launch, its front
 * spreading out from there as the package's default parametric threat runs
 * its own front, at the threat's speed and across the threat's edge, so the
 * air's anomaly reaches a drone `d` meters from the middle as it reaches a
 * point `d` meters behind the threat's western edge. Its heat pushes each drone
 * straight away from the middle, as the front page's plume pushes its birds:
 * from the least heat that pushes a bird, at `FLIGHT.faint`, up to
 * `FLIGHT.push` once the heat reaches 0.42 of the anomaly.
 */
export class Fire {
  readonly edge  = threat.edge_m
  readonly front = new Front()

  /**
   * Reads the share of the anomaly the air holds `d` meters from the middle
   * after `seconds`, one inside the burned ground and zero well clear of it.
   */
  share(seconds: number, d: number): number {
    return this.front.profile(seconds, d)
  }

  /**
   * Reads the distance from the middle at which the air holds `share` of the
   * anomaly after `seconds`, in meters, zero where the fire reaches no further.
   */
  reach(seconds: number, share: number): number {
    return Math.max(0, this.front.at(seconds) + this.edge * Math.atanh(1 - 2 * share))
  }

  /**
   * Reads how far the heat pushes a drone, from zero, below the least heat that
   * pushes one, to one, at full strength, on a square-root scale of the heat
   * between them, since most of the air a drone flies through holds well
   * under a hundredth of the anomaly.
   */
  felt(share: number): number {
    if (share <= FLIGHT.faint) return 0
    return Math.sqrt(Math.min(1, Math.log(share / FLIGHT.faint) / Math.log(0.42 / FLIGHT.faint)))
  }

  /**
   * Writes into `pushes` the push the heat gives each drone of `positions`
   * after `seconds`, around the middle at `east` and `north`.
   */
  push(seconds: number, east: number, north: number, positions: ArrayLike<number>, pushes: Float64Array): void {
    pushes.fill(0)
    for (let at = 0; at < positions.length; at += 3) {
      const dx    = positions[at]! - east
      const dy    = positions[at + 1]! - north
      const d     = Math.max(Math.hypot(dx, dy), 1e-6)
      const share = this.share(seconds, d)
      if (share <= FLIGHT.faint) continue
      const strength  = FLIGHT.push * Math.min(1, share * 2.4) / d
      pushes[at]      = dx * strength
      pushes[at + 1]  = dy * strength
    }
  }
}
