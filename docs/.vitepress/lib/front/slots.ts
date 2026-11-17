import source from '../rendered/composite.json' with { type: 'json' }

/**
 * The default slots and the fields of the plan `mise render` writes out of the
 * package's settings, each slot under its kind.
 */
export type Model = typeof source

/**
 * The roost's disk and height a drawing places, its center east and north and
 * its radius and height in meters.
 */
export type Place = Pick<Model['roost'], 'height_m' | 'radius_m'> & { center_m: [number, number] }

/**
 * The smallest positive single-precision value, which a length divides by in
 * place of zero, as the package's `unit` takes it.
 */
export const TINY = 1.1754944e-38

/**
 * Shortens the horizontal part of the vector at `at` of `vectors` to at most
 * `level`, keeping its direction, and clips its vertical part between `low`
 * and `high`, as the package's `capped` does.
 */
export function capped(vectors: Float64Array, at: number, high: number, level: number, low: number): void {
  const share     = Math.min(1, level / Math.max(Math.hypot(vectors[at]!, vectors[at + 1]!), TINY))
  vectors[at]     = vectors[at]! * share
  vectors[at + 1] = vectors[at + 1]! * share
  vectors[at + 2] = Math.min(Math.max(vectors[at + 2]!, low), high)
}

/**
 * Starts, copies, or counts down the zig `forces` holds at `at` after one step,
 * as the package's `Zig` does, where `turns` holds the roost's acceleration and
 * `before` the command as the step began, each at `at`:
 *
 * - A resting drone the roost turns at least as hard as the zig pushes starts
 *   one away from the rim, due once its reaction time passes
 * - A resting drone whose `draw` falls under the chance its `zigs` zigging
 *   neighbors give starts one toward the side `sides` sums, due once its
 *   reaction and cue times less the radio's delay pass
 * - Any other drone counts its zig down by one step
 */
export function excited(
  forces : Float64Array,
  at     : number,
  before : Float64Array,
  draw   : number,
  model  : Model,
  sides  : number,
  turns  : Float64Array,
  zigs   : number
): void {
  const { noise, plan, radio } = model
  const countdown = forces[at]!
  const start     = noise.refractory_s + noise.side_s + noise.back_s
  const hazard    = -Math.log1p(-noise.branching / plan.neighbors) / (noise.side_s + noise.back_s)
  const resting   = countdown <= 0
  const walled    = resting && Math.hypot(turns[at]!, turns[at + 1]!) >= noise.accel_m_s2
  if (walled) {
    forces[at]     = start + noise.reaction_s
    forces[at + 1] = turns[at + 1]! * before[at]! - turns[at]! * before[at + 1]! >= 0 ? 1 : -1
  } else if (resting && draw < -Math.expm1(-hazard * zigs * plan.step_s)) {
    forces[at]     = start + Math.max(noise.reaction_s + noise.cue_s - radio.delay_s, 0)
    forces[at + 1] = sides >= 0 ? 1 : -1
  } else {
    forces[at] = Math.max(countdown - plan.step_s, 0)
  }
}

/**
 * Writes the roost's acceleration on the drone at `at` of `positions` and
 * `commands` into `into` at the same place: the pull toward the height `place`
 * sets, damped against its climb, and the turn away from the rim of the disk
 * `place` sets that its course would reach, as the package's `Home` gives.
 */
export function home(
  at        : number,
  commands  : Float64Array,
  into      : Float64Array,
  place     : Place,
  positions : Float64Array,
  roost     : Model['roost']
): void {
  const { center_m, height_m, radius_m } = place
  const gx      = commands[at]!, gy = commands[at + 1]!
  const ox      = positions[at]! - center_m[0], oy = positions[at + 1]! - center_m[1]
  const along   = ox * gx + oy * gy
  const squared = gx * gx + gy * gy
  const pace    = Math.sqrt(squared)
  const outside = ox * ox + oy * oy - radius_m ** 2
  const reach   = (Math.sqrt(Math.max(along * along - squared * outside, 0)) - along)
                / Math.max(squared, TINY)
  const rate    = outside < 0 ? roost.turn * pace / Math.max(reach, TINY) : Infinity
  const ground  = Math.max(pace, TINY)
  const hx      = gx / ground, hy = gy / ground
  const share   = ox * hx + oy * hy
  const px      = ox - share * hx, py = oy - share * hy
  const lateral = -Math.min(rate, roost.bound_m_s2) / Math.max(Math.hypot(px, py), TINY)
  into[at]      = lateral * px
  into[at + 1]  = lateral * py
  into[at + 2]  = -roost.stiffness_per_s2 * (positions[at + 2]! - height_m)
                - 2 * Math.sqrt(roost.stiffness_per_s2) * commands[at + 2]!
}

/**
 * Tells whether a drone holding `countdown` pushes to its side or back under
 * the zig `noise` sets.
 */
export const zigging = ({ back_s, refractory_s, side_s }: Model['noise'], countdown: number): boolean =>
  countdown > refractory_s && countdown <= refractory_s + side_s + back_s
