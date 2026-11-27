import type { Geometry }      from '../canvas/part.ts'
import { Tones, translucent } from '../canvas/tones.ts'
import * as common            from '../common.ts'
import source                 from '../rendered/identity.json' with { type: 'json' }
import { visible }            from '../threat/shades.ts'
import type { Player }        from './player.ts'
import type { Burn, Scene }   from './scene.ts'

type Scheme = Geometry['scheme']

const tones = new Tones(source)

/**
 * The air's share of the anomaly at the edge of each band, from a fiftieth at
 * the outer edge to nine tenths at the core.
 */
const SHARES = [0.02, 0.1, 0.3, 0.6, 0.9] as const

/**
 * The colors a drone takes from the flock's own, at no heat felt, to the hot
 * end of the ramp, at full strength, one more than the steps its heat is
 * rounded to.
 */
const STEPS = 32

/**
 * Finds the color `share` of the way along `ramp`.
 */
function along(ramp: readonly string[], share: number): string {
  const place = common.clamp(share) * (ramp.length - 1)
  const below = Math.min(Math.floor(place), ramp.length - 2)
  return common.mix(ramp[below]!, ramp[below + 1]!, place - below)
}

/**
 * The heat ramp's part that holds its contrast on the ground of `scheme`, from
 * cool to hot, as the site's threat plot reads it.
 */
const rampOf = (scheme: Scheme): readonly string[] => visible(tones.heat, scheme)

/**
 * Builds the colors a drone takes in `scheme`, each easing from the flock's
 * own along the ramp as the heat felt rises.
 */
function graded(scheme: Scheme): readonly string[] {
  const base = tones.flock(scheme, 1, 1)
  const ramp = rampOf(scheme)
  return Array.from({ length: STEPS + 1 }, (_, k) => common.mix(base, along(ramp, k / STEPS), Math.min(1, k / STEPS * 2.5)))
}

const GRADED: Record<Scheme, readonly string[]> = { dark: graded('dark'), light: graded('light') }

/**
 * The place the share at a drone is read from.
 */
const at = new Float32Array(3)

/**
 * Reads the share of the anomaly the air holds at the drone at `index` at the
 * moment shown, from its distance to the roost's middle.
 */
function shareAt(player: Player, index: number): number {
  const { fire, roost: { center_m: [east, north] } } = player.flight
  player.place(index, at)
  return fire.share(player.t, Math.hypot(at[0]! - east, at[1]! - north))
}

/**
 * Fills the disk out to where the air holds `share` of the anomaly in `color`
 * at `alpha`. Each drawing of the fire lies at the roost's height, where the
 * flock flies, since the threat's anomaly holds at every height, cooled only
 * by the Exner function, so the drones read against the air they fly through.
 */
function disk(ctx: CanvasRenderingContext2D, scene: Scene, player: Player, share: number, color: string, alpha: number): void {
  const radius = player.flight.fire.reach(player.t, share)
  if (radius <= 0) return
  scene.circle(ctx, radius, player.flight.roost.height_m)
  ctx.fillStyle = translucent(color, alpha)
  ctx.fill()
}

/**
 * The fire as burned air in bands of the ramp, the air's share of the anomaly
 * stepping from a fiftieth at the outer edge to nine tenths at the core, each
 * drone easing along the ramp by how hard the heat pushes it.
 */
export const BANDS: Burn = {
  ground: (ctx, g, scene, player) => {
    const ramp = rampOf(g.scheme)
    const lift = g.scheme === 'dark' ? 1 : 0.85
    for (const [k, share] of SHARES.entries()) disk(ctx, scene, player, share, along(ramp, k / (SHARES.length - 1)), 0.22 * lift)
  },
  said: 'The deeper a band, the hotter the air, and a drone reddens the harder the heat pushes it.',
  tone: (player, index, scheme) => {
    const felt = player.flight.fire.felt(shareAt(player, index))
    return felt > 0 ? GRADED[scheme][Math.round(felt * STEPS)] : undefined
  }
}
