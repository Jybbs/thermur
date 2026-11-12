import { wcagContrast }          from 'culori'
import { describe, expect, it } from 'vitest'

import * as identity             from '../../lib/identity.ts'
import source                    from '../../lib/rendered/identity.json' with { type: 'json' }
import { FLOORS, laid, painted } from '../common/contrast.ts'

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
 * Lists each custom property of `record` whose kind is `kind`, by its name.
 */
const ofKind = <T extends { kind: string }>(record: Record<string, T>, kind: string): [string, T][] =>
  Object.entries(record).filter(([, entry]) => entry.kind === kind)

/**
 * Lists each custom property the theme mixes of `kind`, by its name.
 */
const kinded = (kind: string): [string, identity.Mix][] => ofKind(identity.MIXES, kind)

/**
 * Reads the color `heat` paints under `scheme`, its step of the heat ramp laid
 * over the ground at its share.
 */
const heated = ({ share = 1, step }: identity.Heat, scheme: identity.Scheme): string =>
  laid(identity.shadeOf('ground', scheme), identity.heatRamp(source)[step - 1]!, share)

/**
 * Pairs each of `entries` with each ground it is laid over.
 */
const overGrounds = <T>(entries: [string, T][]): (readonly [string, string, T, identity.Mix])[] =>
  entries.flatMap(([name, entry]) =>
    kinded('ground').map(([ground, under]) => [name, ground, entry, under] as const))

/**
 * Pairs each property holding a floor with each ground it is laid over.
 */
const pairs = overGrounds(checked)

describe.for(SCHEMES)('under the %s scheme', scheme => {
  it.for(Object.entries(identity.ON_LEAD))(
    'the ground holds a text’s contrast as %s on a button filled in %s',
    ([, fill]) => {
      const label = identity.shadeOf('ground', scheme)
      expect(wcagContrast(label, painted(identity.MIXES[fill]!, scheme))).toBeGreaterThanOrEqual(FLOORS.text)
    }
  )

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

describe.for(SCHEMES)('the fill under the %s scheme', scheme => {
  it.for(ofKind(identity.FILL, 'fill'))(
    'the ink holds a text’s contrast against its %s',
    ([, fill]) => {
      expect(wcagContrast(painted(ink, scheme), heated(fill[scheme], scheme))).toBeGreaterThanOrEqual(
        FLOORS.text
      )
    }
  )

  it.for(overGrounds(ofKind(identity.FILL, 'mark')))(
    'its %s holds a mark’s contrast against %s',
    ([, , fill, ground]) => {
      expect(wcagContrast(heated(fill[scheme], scheme), painted(ground, scheme))).toBeGreaterThanOrEqual(
        FLOORS.mark
      )
    }
  )

  it('its edge holds a mark’s contrast against the ember behind it on a link', () => {
    const { '--th-fill-edge': edge, '--th-fill-ember': ember } = identity.FILL
    expect(wcagContrast(heated(edge![scheme], scheme), heated(ember![scheme], scheme))).toBeGreaterThanOrEqual(
      FLOORS.mark
    )
  })
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
  it('sets the label of a button filled in the lead in the ground, at rest, hovered, and pressed', () => {
    const css = identity.identityCss()
    for (const state of ['active-', 'hover-', '']) {
      expect(css).toContain(`--vp-button-brand-${state}text: var(--th-ground);`)
    }
  })

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

  it('sets each fill color to a step of the heat ramp per scheme, the ember laid over the ground', () => {
    const css = identity.identityCss()
    expect(css).toContain('--th-fill-edge: light-dark(var(--th-heat-4), var(--th-heat-1));')
    expect(css).toContain('--th-fill-hot: light-dark(var(--th-heat-2), var(--th-heat-4));')
    expect(css).toContain(
        '--th-fill-ember: light-dark(color-mix(in srgb, var(--th-heat-1) 50%, var(--th-ground)), '
      + 'var(--th-heat-6));'
    )
  })

  it('carries no hex outside the identity’s own colors', () => {
    const declared = new Set(Object.values(source.palette.colors))
    const written  = [...identity.identityCss().matchAll(/#[\da-f]{6}/gu)].map(([hex]) => hex)
    expect(written.filter(hex => !declared.has(hex))).toEqual([])
  })
})

describe('roles', () => {
  it('lists every palette entry holding a color per scheme, sorted by name', () => {
    expect(identity.roles).toEqual(['critical', 'ground', 'ink', 'lead', 'warning'])
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
