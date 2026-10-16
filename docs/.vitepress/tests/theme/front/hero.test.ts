// @vitest-environment happy-dom
import { mount }              from '@vue/test-utils'
import { expect, it, vi }     from 'vitest'
import { defineComponent, h } from 'vue'

import source    from '../../../lib/rendered/identity.json' with { type: 'json' }
import FrontHero from '../../../theme/front/hero.ts'

vi.mock('vitepress/theme-without-fonts', () => ({
  VPImage: defineComponent({
    props: { image: { required: true, type: Object } },
    setup: ({ image }) => () =>
      h('img', { 'alt': image.alt, 'data-dark': image.dark, 'data-light': image.light })
  })
}))

it('sets the wordmark in each scheme as the page’s heading over the tagline', () => {
  const hero  = mount(FrontHero)
  const image = hero.find('h1 img')
  expect(image.attributes()).toMatchObject({
    'alt'        : 'Thermur',
    'data-dark'  : `/${source.marks.wordmark.dark}`,
    'data-light' : `/${source.marks.wordmark.light}`
  })
  expect(hero.find('p').text()).toBe(source.tagline.text)
})
