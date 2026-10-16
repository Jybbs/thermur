import * as common                 from './common.ts'
import { type Identity, heatRamp } from './identity.ts'
import { CANVAS }                  from './layout.ts'
import * as scene                  from './scene.ts'

/**
 * A stretch of the fire's front as `frontPath` traces it, from `from` to `to`
 * radians from the downwind direction in `steps` steps, drawn in to `share` of
 * the front's reach and moved `shift` pixels downwind.
 */
interface Arc {
  from  : number
  share : number
  shift : number
  steps : number
  to    : number
}

/**
 * A fire seen from above as `fireground` draws it: its middle at `x` and `y`,
 * scaled by `scale`, stretched `along` the wind and `across` it with the wind
 * blowing toward the card's right edge, its embers drawn from `seed`, and its
 * smoke trailing downwind in `puffs` puffs.
 */
export interface Fire {
  across : number
  along  : number
  puffs  : number
  scale  : number
  seed   : number
  x      : number
  y      : number
}

/**
 * A fire seen from above drawn as SVG, meaning the scar it has burned, its
 * front, and its smoke, beside two readings of a point on the card: `gap`, how
 * deep in the fire's heat or its smoke the point sits, 2 inside either, where
 * no bird flies, and outside them rising from nothing to about one half at the
 * edge of either, and `near`, how near the fire's middle it stands, from 1 at
 * the middle to 0 well clear of it.
 */
export interface Fireground {
  gap  : (x: number, y: number) => number
  near : (x: number, y: number) => number
  svg  : string
}

/**
 * One puff of smoke, an ellipse at `x` and `y` reaching `rx` along the wind and
 * `ry` across it.
 */
interface Puff {
  rx : number
  ry : number
  x  : number
  y  : number
}

/**
 * The tones a fire seen from above draws in that the palette names no color
 * for, the shadow at the middle of the scar and the smoke.
 */
const SHADE = { scar: '#0b090a', smoke: '#3b3437' } as const
/**
 * The terrain under the fire, `count` contour lines from `top` pixels down,
 * `spacing` pixels apart, every fourth line brighter than the rest.
 */
const TERRAIN = { count: 12, spacing: 52, top: 30 } as const
/**
 * The whole of the fire's front at its full reach, traced in 240 steps.
 */
const WHOLE = { from: 0, share: 1, shift: 0, steps: 240, to: Math.PI * 2 } as const
/**
 * The blurs the fire's glow and its smoke are drawn through, `b4` and `b12`,
 * Gaussian blurs with standard deviations of 4 and 12 pixels over a region
 * reaching past the shape each blurs.
 */
export const blurs = scene.blur('b4', 4, 500) + scene.blur('b12', 12, 500)

/**
 * Lists the embers speckling the scar, each at a random place inside 95% of the
 * front's reach, drawn in one of the middle steps of the heat ramp.
 */
function embers(fire: Fire, heat: readonly string[]): string {
  const draw = common.mulberry(fire.seed)
  return Array.from({ length: Math.round(160 * fire.scale ** 2) }, () => {
    const [x, y]  = frontPoint(fire, draw() * Math.PI * 2, Math.sqrt(draw()) * 0.95)
    const radius  = scene.tenths(0.6 + draw() * 1.2)
    const fill    = heat[2 + Math.floor(draw() * 3)]
    const opacity = scene.tenths((0.25 + draw() * 0.5) * 100) / 100
    return `<circle cx="${scene.tenths(x)}" cy="${scene.tenths(y)}" r="${radius}" fill="${fill}" `
         + `fill-opacity="${opacity}"/>`
  }).join('')
}

/**
 * Measures the radius of the fire's front at `angle` radians from the downwind
 * direction, in pixels before the fire is scaled, the front reaching farther
 * downwind than upwind.
 */
function frontAt(angle: number): number {
  return 100
    + 16 * Math.sin(angle * 3 + 1)
    + 10 * Math.sin(angle * 5 + 2.3)
    + 6 * Math.sin(angle * 11 + 0.7)
    + 4 * Math.sin(angle * 19)
    + 30 * Math.cos(angle)
}

/**
 * Traces the stretch of the fire's front `arc` names as an SVG path, the whole
 * of the front at its full reach where `arc` names nothing else.
 */
function frontPath(fire: Fire, arc: Partial<Arc> = {}): string {
  const { from, share, shift, steps, to } = { ...WHOLE, ...arc }
  const points                            = Array.from({ length: steps + 1 }, (_, index) =>
    frontPoint(fire, from + (to - from) * index / steps, share, shift))
  return `M${scene.polyline(points)}`
}

/**
 * Finds the point on the fire's front at `angle`, drawn in to `share` of its
 * reach and moved `shift` pixels downwind.
 */
function frontPoint({ across, along, scale, x, y }: Fire, angle: number, share = 1, shift = 0): common.Pair {
  const radius = frontAt(angle) * scale * share
  return [x + Math.cos(angle) * radius * along + shift, y + Math.sin(angle) * radius * across]
}

/**
 * Lists the puffs of smoke trailing downwind from the fire, each wider and
 * farther up the card than the one before it.
 */
function puffsOf({ puffs, scale, x, y }: Fire): Puff[] {
  return Array.from({ length: puffs }, (_, index) => ({
    rx : (50 + index * 13) * scale,
    ry : (24 + index * 7) * scale,
    x  : x + (120 + index * 32) * scale,
    y  : y - (30 + index * 15) * scale * 0.6
  }))
}

/**
 * Traces the terrain under the fire as contour lines across the card in
 * `color`, every fourth line brighter than the rest.
 */
export function contours(color: string, terrain: typeof TERRAIN = TERRAIN): string {
  return Array.from({ length: terrain.count }, (_, k) => {
    const height = (x: number): number => terrain.top + k * terrain.spacing
      + 20 * Math.sin(x / 130 + k * 0.7) + 9 * Math.sin(x / 47 + k) + 4 * Math.sin(x / 19 + k * 2)
    const opacity = k % 4 === 0 ? 0.14 : 0.08
    return `<path d="${scene.trace(CANVAS.width, height)}" fill="none" stroke="${color}" `
         + `stroke-opacity="${opacity}"/>`
  }).join('')
}

/**
 * Draws `fire` seen from above in the identity's heat ramp: the scar of what
 * has burned speckled with embers and shadowed at its middle, the smoke
 * trailing downwind, and the front glowing along its downwind edge, beside the
 * readings of how deep in its heat a point sits and how near its middle it
 * stands. The first 70% of the puffs part the flock as the heat does.
 */
export function fireground(fire: Fire, identity: Identity): Fireground {
  const heat   = heatRamp(identity)
  const smoke  = puffsOf(fire)
  const plumes = smoke.slice(0, Math.ceil(fire.puffs * 0.7))
  const edge   = frontPath(fire, { from: -1.05, steps: 150, to: 1.25 })
  const glow   = (color: string, width: number, extra: string): string =>
    `<path d="${edge}" fill="none" stroke="${color}" stroke-width="${scene.tenths(width)}"${extra}/>`
  const puffs = smoke.map(({ rx, ry, x, y }, index) =>
      `<ellipse cx="${scene.tenths(x)}" cy="${scene.tenths(y)}" rx="${scene.tenths(rx)}" `
    + `ry="${scene.tenths(ry)}" fill="${SHADE.smoke}" `
    + `fill-opacity="${scene.tenths((0.24 - index * 0.016) * 100) / 100}"/>`)

  return {
    gap: (x, y) => {
      const [lx, ly] = [(x - fire.x) / fire.along, (y - fire.y) / fire.across]
      const reach    = Math.hypot(lx, ly) / (frontAt(Math.atan2(ly, lx)) * fire.scale + 70)
      let depth      = scene.depth(reach, 1, 1.5)
      for (const puff of plumes) {
        const inside = Math.hypot((x - puff.x) / puff.rx, (y - puff.y) / puff.ry)
        depth        = Math.max(depth, scene.depth(inside, 0.7, 1.25))
      }
      return depth
    },

    near: (x, y) => {
      const reach = Math.hypot((x - fire.x) / fire.along, (y - fire.y) / fire.across) / (fire.scale * 320)
      return common.clamp(1 - reach)
    },

    svg: [
      `<path d="${frontPath(fire)}Z" fill="${heat[5]}" fill-opacity="0.5"/>`,
      `<path d="${frontPath(fire, { share: 0.62, shift: -26 * fire.scale })}Z" fill="${SHADE.scar}" `
    + 'fill-opacity="0.55" filter="url(#b12)"/>',
      embers(fire, heat),
      `<g filter="url(#b12)">${puffs.join('')}</g>`,
      glow(heat[2]!, 26 * Math.sqrt(fire.scale), ' stroke-opacity="0.5" filter="url(#b12)"'),
      glow(heat[1]!, 6 * Math.sqrt(fire.scale), ' filter="url(#b4)"'),
      glow(heat[0]!, 1.8, '')
    ].join('')
  }
}
