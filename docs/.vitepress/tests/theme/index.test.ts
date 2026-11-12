import { describe, expect, it, vi } from 'vitest'
import type { VNode }               from 'vue'

import theme from '../../theme/index.ts'

vi.mock('vitepress/theme-without-fonts', () => ({ default: { Layout: 'Layout' } }))
vi.mock('../../theme/crop.ts', () => ({ default: 'CropKept' }))
vi.mock('../../theme/identity/faces.ts', () => ({ default: 'IdentityFaces' }))
vi.mock('../../theme/identity/heat.ts', () => ({ default: 'IdentityHeat' }))
vi.mock('../../theme/identity/marks.ts', () => ({ default: 'IdentityMarks' }))
vi.mock('../../theme/identity/palette.ts', () => ({ default: 'IdentityPalette' }))
vi.mock('../../theme/front/hero.ts', () => ({ default: 'FrontHero' }))
vi.mock('../../theme/front/sky.ts', () => ({ default: 'FrontSky' }))
vi.mock('../../theme/not-found/page.ts', () => ({ default: 'NotFoundPage' }))
vi.mock('../../theme/pause.ts', () => ({ default: 'MotionPause' }))
vi.mock('../../theme/rules/neighbors.ts', () => ({ default: 'RulesNeighbors' }))
vi.mock('../../theme/rules/reach.ts', () => ({ default: 'RulesReach' }))
vi.mock('../../theme/rules/spacing.ts', () => ({ default: 'RulesSpacing' }))
vi.mock('../../theme/threat.ts', () => ({ default: 'ThreatFront' }))

describe('the theme', () => {
  it('extends the default theme without its faces', () => {
    expect(theme.extends).toEqual({ Layout: 'Layout' })
  })

  it('registers the widgets a page’s rules write by name', () => {
    const component = vi.fn<(name: string, widget: unknown) => void>()
    theme.enhanceApp({ app: { component } } as unknown as Parameters<typeof theme.enhanceApp>[0])
    expect(component.mock.calls).toEqual([
      ['CropKept', 'CropKept'],
      ['IdentityFaces', 'IdentityFaces'],
      ['IdentityHeat', 'IdentityHeat'],
      ['IdentityMarks', 'IdentityMarks'],
      ['IdentityPalette', 'IdentityPalette'],
      ['RulesNeighbors', 'RulesNeighbors'],
      ['RulesReach', 'RulesReach'],
      ['RulesSpacing', 'RulesSpacing'],
      ['ThreatFront', 'ThreatFront']
    ])
  })

  it('fills the layout’s slots with the front page’s pieces, the pause, and the lost page', () => {
    const layout = theme.Layout() as VNode
    const slots  = layout.children as Record<string, () => VNode>
    expect(layout.type).toBe('Layout')
    expect(Object.fromEntries(Object.entries(slots).map(([name, slot]) => [name, slot().type]))).toEqual({
      'home-hero-before'      : 'FrontSky',
      'home-hero-info'        : 'FrontHero',
      'nav-bar-content-after' : 'MotionPause',
      'not-found'             : 'NotFoundPage'
    })
  })
})
