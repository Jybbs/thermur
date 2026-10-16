import { interpolate, wcagContrast } from 'culori'
import { describe, expect, it }      from 'vitest'

import * as identity from '../../lib/identity.ts'
import source        from '../../lib/rendered/identity.json' with { type: 'json' }

/**
 * The contrast WCAG 2.2 sets for text and for a mark that carries meaning.
 */
const FLOORS = { mark: 3, text: 4.5 } as const
/**
 * The two schemes every contrast holds under.
 */
const SCHEMES = ['dark', 'light'] as const
/**
 * The custom property the body text is set in.
 */
const ink = identity.MIXES['--vp-c-text-1']!
/**
 * Each custom property whose kind holds a contrast floor, by its name.
 */
const checked = Object.entries(identity.MIXES).filter(([, mix]) => mix.kind in FLOORS)

/**
 * Calls one of a plugin's hooks on `id`, as Vite calls it.
 */
const call = <T>(hook: unknown, id: string): T => (hook as (id: string) => T)(id)

/**
 * Lists each custom property of `kind`, by its name.
 */
const kinded = (kind: string): [string, identity.Mix][] =>
  Object.entries(identity.MIXES).filter(([, mix]) => mix.kind === kind)

/**
 * Reads the color `mix` paints under `scheme`, its role laid over the role
 * beneath it at its share, as `color-mix` in sRGB lays it.
 */
const painted = ({ over = 'ground', role, share }: identity.Mix, scheme: identity.Scheme): string => {
  const { palette } = source
  const colors      = [palette[over][scheme], palette[role][scheme]].map(name => identity.colorOf(name))
  return interpolate(colors, 'rgb')(share) as unknown as string
}

/**
 * Pairs each property holding a floor with each ground it is laid over.
 */
const pairs = checked.flatMap(([name, mix]) => kinded('ground').map(([ground, under]) =>
  [name, ground, mix, under] as const))

describe.for(SCHEMES)('under the %s scheme', scheme => {
  it.for(pairs)('%s holds its contrast against %s', ([, , mix, ground]) => {
    expect(wcagContrast(painted(mix, scheme), painted(ground, scheme))).toBeGreaterThanOrEqual(
      FLOORS[mix.kind as keyof typeof FLOORS]
    )
  })

  it.for(kinded('fill'))('the ink holds a text’s contrast against the surface %s', ([, fill]) => {
    expect(wcagContrast(painted(ink, scheme), painted(fill, scheme))).toBeGreaterThanOrEqual(FLOORS.text)
  })

  it.for(source.palette.series.map((pair, index) => [index + 1, pair] as const))(
    'chart series %i holds a mark’s contrast against the ground',
    ([, pair]) => {
      const ground = identity.colorOf(source.palette.ground[scheme])
      expect(wcagContrast(identity.colorOf(pair[scheme]), ground)).toBeGreaterThanOrEqual(FLOORS.mark)
    }
  )
})

describe('colorOf', () => {
  it('reads a color the palette names', () => {
    expect(identity.colorOf('woodsmoke')).toBe('#17171b')
  })

  it('names a color the palette lacks', () => {
    const missing = 'the palette declares no color named no-such-color'
    expect(() => identity.colorOf('no-such-color')).toThrow(missing)
  })
})

describe('fontsHref', () => {
  it('loads every face at its weights, roman and italic where a face takes both', () => {
    expect(identity.fontsHref()).toBe(
        'https://fonts.googleapis.com/css2'
      + '?family=Crimson+Pro:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700'
      + '&family=Bellefair'
      + '&family=JetBrains+Mono:wght@400;500&display=swap'
    )
  })

  it('keeps the italic of a face set at 400 alone', () => {
    const faces = { ...source.faces, display: { ...source.faces.display, italic: true } }
    expect(identity.fontsHref({ ...source, faces })).toContain('&family=Bellefair:ital,wght@0,400;1,400&')
  })
})

describe('identityCss', () => {
  it('sets a role through light-dark only where its two colors differ', () => {
    const css = identity.identityCss()
    expect(css).toContain('--th-ground: light-dark(#f2f1ee, #17171b);')
    expect(css).toContain('--th-warning: #e8c840;')
  })

  it('lays each mixed property over the ground and sets each face’s stack', () => {
    const css = identity.identityCss()
    expect(css).toContain('--vp-c-text-2: color-mix(in srgb, var(--th-ink) 78%, var(--th-ground));')
    expect(css).toContain('--vp-c-text-1: var(--th-ink);')
    expect(css).toContain('--vp-c-brand-2: color-mix(in srgb, var(--th-lead) 80%, var(--th-ink));')
    expect(css).toContain("--th-font-display: 'Bellefair', Didot, Georgia, serif;")
  })

  it('carries no hex outside the identity’s own colors', () => {
    const declared = new Set(Object.values(source.palette.colors))
    const written  = [...identity.identityCss().matchAll(/#[\da-f]{6}/gu)].map(([hex]) => hex)
    expect(written.filter(hex => !declared.has(hex))).toEqual([])
  })
})

describe('identityModule', () => {
  const plugin = identity.identityModule()

  it('serves the stylesheet under its virtual id', () => {
    const resolved = call<string>(plugin.resolveId, 'virtual:identity.css')
    expect(call(plugin.load, resolved)).toBe(identity.identityCss())
  })

  it('leaves every other id to the next plugin', () => {
    expect(call(plugin.resolveId, './theme.css')).toBeUndefined()
    expect(call(plugin.load, './theme.css')).toBeUndefined()
  })
})
