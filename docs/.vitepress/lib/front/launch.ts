import type { Model } from './slots.ts'

/**
 * Launches a flock around the plan's center as the package's plan launches
 * one, in a Gaussian cloud whose length, width, and depth the plan's axes set
 * along the course, across it, and up, holding the plan's spacing cubed for
 * each drone, its height raised to `clearance_sd` deviations of its depth
 * where that clears the center's and any drone drawn under the ground set on
 * it, each flying the course within its scatter at the cruise speed and
 * climbing where it was drawn under the ground, every draw from `normal`.
 * Each drone's east, north, and height go to three places of `positions`, and
 * its velocity likewise to `velocities`.
 */
export function launched(
  normal     : () => number,
  plan       : Model['plan'],
  positions  : Float64Array,
  velocities : Float64Array
): void {
  const { axes, center_m, clearance_sd, course_deg, cruise_m_s, scatter_deg, spacing_m } = plan
  const course  = course_deg * Math.PI / 180
  const scatter = scatter_deg * Math.PI / 180
  const cloud   = 4 * Math.PI * axes[0]! * axes[1]! * axes[2]! / 3
  const scale   = spacing_m * Math.cbrt(positions.length / 3 / cloud)
  const depth   = axes[2]! * scale
  const height  = Math.max(center_m[2]!, clearance_sd * depth)
  const cos     = Math.cos(course)
  const sin     = Math.sin(course)
  for (let at = 0; at < positions.length; at += 3) {
    const along       = normal() * axes[0]! * scale
    const across      = normal() * axes[1]! * scale
    positions[at]     = center_m[0]! + along * cos - across * sin
    positions[at + 1] = center_m[1]! + along * sin + across * cos
    positions[at + 2] = Math.max(height + normal() * depth, 0)
  }
  for (let at = 0; at < velocities.length; at += 3) {
    const x     = cos + normal() * scatter
    const y     = sin + normal() * scatter
    const z     = normal() * scatter
    const speed = cruise_m_s / Math.hypot(x, y, z)
    velocities[at]     = x * speed
    velocities[at + 1] = y * speed
    velocities[at + 2] = (positions[at + 2] === 0 ? Math.abs(z) : z) * speed
  }
}
