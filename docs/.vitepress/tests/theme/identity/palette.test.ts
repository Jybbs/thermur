// @vitest-environment happy-dom
import { enableAutoUnmount, mount }            from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createSSRApp, ref }                   from 'vue'
import { renderToString }                      from 'vue/server-renderer'

import { colorOf, roles } from '../../../lib/identity.ts'
import source             from '../../../lib/rendered/identity.json' with { type: 'json' }
import IdentityPalette    from '../../../theme/identity/palette.ts'

const isDark = ref(false)

vi.mock('vitepress', () => ({ useData: () => ({ isDark }) }))

enableAutoUnmount(afterEach)

afterEach(() => {
  isDark.value = false
})

describe('IdentityPalette', () => {
  it('draws a line and a switch per chart series, every one on', () => {
    const widget = mount(IdentityPalette)
    expect(widget.findAll('.identity-line').map(line => line.attributes('data-shown')))
      .toEqual(source.palette.series.map(() => 'true'))
    expect(widget.findAll('[role="switch"]').map(button => button.text()))
      .toEqual(source.palette.series.map((_, place) => `Chart ${place + 1}`))
  })

  it('lifts a series’s line off and draws it back on from its switch', async () => {
    const widget = mount(IdentityPalette)
    const toggle = widget.findAll('[role="switch"]')[2]!
    await toggle.trigger('click')
    expect([toggle.attributes('aria-checked'), widget.findAll('.identity-line')[2]!.attributes('data-shown')])
      .toEqual(['false', 'false'])
    await toggle.trigger('click')
    expect(toggle.attributes('aria-checked')).toBe('true')
  })

  it('writes each role’s hex under both schemes, keyed in the table’s order', () => {
    const widget = mount(IdentityPalette)
    const keys   = widget.findAll('.identity-key')
    expect(keys.map(key => key.get('.identity-caps').text())).toEqual(roles)
    const lead   = keys[roles.indexOf('lead')]!
    expect(lead.get('.identity-on-light').text()).toBe(colorOf(source.palette.lead.light))
    expect(lead.get('.identity-on-dark').text()).toBe(colorOf(source.palette.lead.dark))
  })

  it('labels its sketch as made up and renders on the server', async () => {
    const html = await renderToString(createSSRApp(IdentityPalette))
    expect(html).toContain('A sketch, its temperatures made up')
    expect(html).toContain('role="img"')
  })
})

describe('copying from IdentityPalette', () => {
  it.for([
    [false, 'light'], [true, 'dark']
  ] as const)('copies the hex of the scheme on display, dark: %s', async ([dark, scheme]) => {
    isDark.value = dark
    const widget = mount(IdentityPalette, { attachTo: document.body })
    await widget.findAll('.identity-copy')[1]!.trigger('click')
    await vi.waitFor(async () =>
      expect(await navigator.clipboard.readText()).toBe(colorOf(source.palette.series[1]![scheme])))
    await vi.waitFor(() => expect(widget.get('[aria-live]').text()).toMatch(/^Copied #/u))
  })

  it('copies a role’s hex from its key in the scheme on display', async () => {
    isDark.value = true
    const widget = mount(IdentityPalette, { attachTo: document.body })
    await widget.findAll('.identity-key')[roles.indexOf('lead')]!.trigger('click')
    await vi.waitFor(async () =>
      expect(await navigator.clipboard.readText()).toBe(colorOf(source.palette.lead.dark)))
  })
})
