import { type Tones, translucent } from '../canvas/tones.ts'
import { mix }                     from '../common.ts'
import { type Identity, type Scheme, shadeOf } from '../identity.ts'

/**
 * The colors a plot draws in for one scheme: the area and the stroke of the
 * air, the colors a cell takes from cool to hot, the lit ground beneath them,
 * and the page's ground, ink, and lead.
 */
export interface Shades {
  area   : CanvasGradient
  cells  : readonly string[]
  ember  : string
  ground : string
  ink    : string
  lead   : string
  rule   : string
  stroke : CanvasGradient
  word   : string
}

/**
 * The steps of the table of colors a cell takes.
 */
export const STEPS = 24

/**
 * Finds the color `share` of the way from the first of `ramp` to its last.
 */
function along(ramp: readonly string[], share: number): string {
  const place = share * (ramp.length - 1)
  const below = Math.min(Math.floor(place), ramp.length - 2)
  return mix(ramp[below]!, ramp[below + 1]!, place - below)
}

/**
 * Builds the colors a plot draws in for `scheme`, the gradients running down
 * the canvas from `top` to `foot` for the air's stroke and to `y1` for its
 * area.
 *
 * Args:
 *   ctx      : The canvas the gradients are built on.
 *   foot     : The pixel the stroke's gradient ends on, at the coolest air.
 *   identity : The identity the page's colors are read from.
 *   scheme   : The scheme the colors are built for.
 *   tones    : The fill strings the heat ramp and the ink are read from.
 *   top      : The pixel both gradients start on, at the hottest air.
 *   y1       : The pixel the area's gradient ends on, at the plot's foot.
 */
export function shadesOf(
  ctx      : CanvasRenderingContext2D,
  foot     : number,
  identity : Identity,
  scheme   : Scheme,
  tones    : Tones,
  top      : number,
  y1       : number
): Shades {
  const ramp   = visible(tones.heat, scheme)
  const stroke = ctx.createLinearGradient(0, top, 0, foot)
  const area   = ctx.createLinearGradient(0, top, 0, y1)
  const lift   = scheme === 'dark' ? 0.34 : 0.26
  for (const [index, color] of ramp.toReversed().entries()) {
    const at = index / (ramp.length - 1)
    stroke.addColorStop(at, color)
    area.addColorStop(at, translucent(color, lift * (1 - at) ** 1.5))
  }
  return {
    area   : area,
    cells  : Array.from({ length: STEPS + 1 }, (_, step) => along(ramp, step / STEPS)),
    ember  : ramp[1]!,
    ground : shadeOf('ground', scheme, identity),
    ink    : tones.ink(scheme),
    lead   : shadeOf('lead', scheme, identity),
    rule   : tones.ink(scheme, 0.09),
    stroke : stroke,
    word   : tones.ink(scheme, 0.72)
  }
}

/**
 * Reads the part of the heat ramp that holds a mark's contrast on the ground of
 * `scheme`, from cool to hot: the three hottest on the dark ground, and the
 * three of the middle on the light one.
 */
export const visible = (heat: readonly string[], scheme: Scheme): readonly string[] =>
  (scheme === 'dark' ? heat.slice(0, 3) : heat.slice(2, 5)).toReversed()
