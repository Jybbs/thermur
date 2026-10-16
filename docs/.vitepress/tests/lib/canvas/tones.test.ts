import { describe, expect, it } from 'vitest'

import { Tones, translucent } from '../../../lib/canvas/tones.ts'
import { mix }                from '../../../lib/common.ts'
import { colorOf }            from '../../../lib/identity.ts'
import source                 from '../../../lib/rendered/identity.json' with { type: 'json' }

/**
 * The palette the identity declares, which every case reads its colors from.
 */
const { palette } = source
/**
 * The tones every case reads, drawn in the identity the site ships.
 */
const tones = new Tones()

/**
 * Reads the hex of the palette color filed under `name`.
 */
const named = (name: string): string => colorOf(name)

describe('the sky', () => {
  it('takes the dawn in the light scheme, its cream warming to a peach over a deep warm ridge', () => {
    const sky = tones.sky('light')
    expect(sky.top).toBe(named(palette.ground.light))
    expect(sky.bands.at(-1)).toBe(mix(named(palette.ground.light), named(palette.heat[0]!), 0.42))
    expect(sky.ridge).toBe(mix(named(palette.ink.light), named(palette.heat[5]!), 0.22))
    expect(sky.shade).toBe(named(palette.ink.light))
  })

  it('takes the settled dusk in the dark scheme, its night warming toward the fire at the horizon', () => {
    const sky = tones.sky('dark')
    const lit = mix(named(palette.ground.dark), named(palette.series[0]!.dark), 0.12)
    expect(sky.top).toBe(named(palette.ground.dark))
    expect(sky.bands.at(-1)).toBe(mix(lit, named(palette.ground.light), 0.035))
    expect(sky.ridge).toBe(mix(named(palette.ground.dark), '#000000', 0.32))
    expect(sky.shade).toBe(mix(named(palette.ground.dark), '#000000', 0.48))
    expect(sky.solid).toBe(0)
  })

  it('writes each scheme’s fills once, returning the same record on every read', () => {
    expect(tones.sky('dark')).toBe(tones.sky('dark'))
    expect(tones.sky('light')).not.toBe(tones.sky('dark'))
  })
})

describe('the flock and the ink', () => {
  it('runs the flock from the scheme’s ink at its foot to the fire’s color at its head', () => {
    expect(tones.flock('light', 0, 1)).toBe(named(palette.ink.light))
    expect(tones.flock('dark', 1, 1)).toBe(named(palette.series[0]!.dark))
    expect(tones.flock('dark', 2, 1)).toBe(tones.flock('dark', 1, 1))
  })

  it('writes the ink at an opacity as an rgb() color and opaque as its hex', () => {
    expect(tones.ink('dark', 0.1)).toMatch(/^rgb\(\d+ \d+ \d+ \/ 0\.1\)$/u)
    expect(tones.ink('dark')).toBe(named(palette.ink.dark))
  })

  it('writes the flock and the ink once per scheme and opacity, reading the palette no more after', () => {
    let reads     = 0
    const watched = new Tones({
      ...source,
      get palette() {
        reads++
        return source.palette
      }
    })
    watched.flock('light', 0.5, 0.92)
    watched.ink('dark', 0.1)
    const first = reads
    watched.flock('light', 0.2, 0.92)
    watched.ink('dark', 0.1)
    expect(reads).toBe(first)
  })
})

describe('the smaller pieces', () => {
  it('writes an opaque color as its hex and a translucent one with its alpha', () => {
    expect(translucent('#102030', 1)).toBe('#102030')
    expect(translucent('#102030', 0.5)).toBe('rgb(16 32 48 / 0.5)')
  })
})
