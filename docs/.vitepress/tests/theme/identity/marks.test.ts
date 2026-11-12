// @vitest-environment happy-dom
import { enableAutoUnmount, mount }            from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { shadeOf }   from '../../../lib/identity.ts'
import source        from '../../../lib/rendered/identity.json' with { type: 'json' }
import IdentityMarks from '../../../theme/identity/marks.ts'

vi.mock('vitepress', () => ({ withBase: (path: string) => `/base${path}` }))

enableAutoUnmount(afterEach)

describe('IdentityMarks', () => {
  it('draws a card per mark, each file on the ground of its scheme, saved on a press', () => {
    const sides = mount(IdentityMarks).findAll('.identity-side')
    const files = Object.values(source.marks).flatMap(pair => [pair.light, pair.dark])
    expect(sides.map(side => side.attributes('download'))).toEqual(files)
    expect(sides.map(side => side.attributes('href'))).toEqual(files.map(file => `/base/${file}`))
    expect((sides[1]!.element as HTMLElement).style.backgroundColor).toBe(shadeOf('ground', 'dark'))
    expect((mount(IdentityMarks).get('.identity-side-light').element as HTMLElement).style.color)
      .toBe(shadeOf('ink', 'light'))
  })

  it('sets dusk across every card from its slider, opening at the middle', async () => {
    const widget = mount(IdentityMarks)
    const figure = widget.get('figure').element as HTMLElement
    expect(figure.style.getPropertyValue('--th-dusk')).toBe('50%')
    const slider = widget.get('input')
    slider.element.value = '20'
    await slider.trigger('input')
    expect(figure.style.getPropertyValue('--th-dusk')).toBe('20%')
  })
})
