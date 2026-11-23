// @vitest-environment happy-dom
import { enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, expect, it }    from 'vitest'
import { h }                        from 'vue'

import CreditsSection from '../../../theme/credits/section.ts'

enableAutoUnmount(afterEach)

it('sets the page’s own lines beside the seal', () => {
  const section = mount(CreditsSection, { slots: { default: () => h('ul', [h('li', 'a line')]) } })
  expect([...section.element.children].map(child => child.getAttribute('class'))).toEqual(['credits-seal', 'credits-lines'])
  expect(section.get('.credits-lines li').text()).toBe('a line')
})

it('stands the seal alone where the page gives it no lines', () => {
  expect(mount(CreditsSection).get('.credits-lines').text()).toBe('')
})
