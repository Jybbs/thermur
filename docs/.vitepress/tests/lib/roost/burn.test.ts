// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { BANDS }         from '../../../lib/roost/burn.ts'
import { Fire }          from '../../../lib/roost/fire.ts'
import { Player }        from '../../../lib/roost/player.ts'
import type { Scene }    from '../../../lib/roost/scene.ts'
import { browser }       from '../../common/browser.ts'
import { geometry }      from '../../common/front.ts'

/**
 * A player standing `seconds` into a flight whose 200 drones stand in a line
 * east from the roost's middle, 5 m apart, and the stand-in pen and scene it
 * draws on, the scene noting the radius of each disk traced.
 */
function drawn(seconds: number, scheme: 'dark' | 'light') {
  const stand  = browser()
  const player = {
    flight : { count: 200, fire: new Fire(), roost: { center_m: [0, 0], height_m: 100 } },
    place  : (index: number, at: Float32Array) => at.set([index * 5, 0, 100]),
    t      : seconds
  } as unknown as Player
  const radii  : number[] = []
  const scene  = { circle: vi.fn<(ctx: unknown, radius: number) => number>((_, radius) => radii.push(radius)) } as unknown as Scene
  BANDS.ground(stand.pen, geometry({ scheme }), scene, player)
  return { player, radii }
}

describe('ground', () => {
  it('draws one disk for each band, the hottest the smallest', () => {
    const { radii } = drawn(200, 'dark')
    expect(radii).toHaveLength(5)
    expect(radii).toEqual(radii.toSorted((a, b) => b - a))
  })

  it('draws no disk the fire has not yet spread to', () => {
    const { radii } = drawn(0, 'light')
    expect(radii.length).toBeLessThan(5)
  })
})

describe('tone', () => {
  it('leaves a drone the heat does not reach its own color', () => {
    const { player } = drawn(0, 'dark')
    const far = Array.from({ length: player.flight.count }, (_, index) => BANDS.tone(player, index, 'dark'))
    expect(far.some(tone => tone === undefined)).toBe(true)
  })

  it('reddens a drone the heat pushes, a color apiece by scheme', () => {
    const { player } = drawn(200, 'dark')
    const dark  = Array.from({ length: player.flight.count }, (_, index) => BANDS.tone(player, index, 'dark')).filter(Boolean)
    const light = Array.from({ length: player.flight.count }, (_, index) => BANDS.tone(player, index, 'light')).filter(Boolean)
    expect(dark.length).toBeGreaterThan(0)
    expect(dark).not.toEqual(light)
  })
})
