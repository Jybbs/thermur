import { describe, expect, it, vi } from 'vitest'
import type { VNode }               from 'vue'

import theme from '../../theme/index.ts'

vi.mock('vitepress/theme-without-fonts', () => ({ default: { Layout: 'Layout' } }))
vi.mock('../../theme/front/hero.ts', () => ({ default: 'FrontHero' }))
vi.mock('../../theme/front/sky.ts', () => ({ default: 'FrontSky' }))
vi.mock('../../theme/not-found/page.ts', () => ({ default: 'NotFoundPage' }))
vi.mock('../../theme/pause.ts', () => ({ default: 'MotionPause' }))

describe('the theme', () => {
  it('extends the default theme without its faces', () => {
    expect(theme.extends).toEqual({ Layout: 'Layout' })
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
