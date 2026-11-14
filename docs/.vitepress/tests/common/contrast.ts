import { interpolate } from 'culori'

import * as identity from '../../lib/identity.ts'

/**
 * The contrast WCAG 2.2 sets for text and for a mark that carries meaning.
 */
export const FLOORS = { mark: 3, text: 4.5 } as const

/**
 * Reads the color `top` paints at `share` over `under`, as `color-mix` in sRGB
 * lays it.
 */
export const laid = (under: string, top: string, share: number): string =>
  interpolate([under, top], 'rgb')(share) as unknown as string

/**
 * Reads the color `mix` paints under `scheme`, its role laid over the role
 * beneath it at its share.
 */
export const painted = (
  { over = 'ground', role, share }: Pick<identity.Mix, 'over' | 'role' | 'share'>,
  scheme: identity.Scheme
): string =>
  laid(identity.shadeOf(over, scheme), identity.shadeOf(role, scheme), share)

/**
 * Reads the color the custom property set for `property` in the inline style
 * `style` paints under `scheme`, a property the theme mixes at its share and a
 * role's own property at full strength.
 */
export function paintedIn(style: string, property: string, scheme: identity.Scheme): string {
  const name = new RegExp(`${property}: var\\((--[\\w-]+)\\)`, 'u').exec(style)![1]!
  return painted(identity.MIXES[name] ?? { role: name.replace('--th-', '') as identity.Role, share: 1 }, scheme)
}
