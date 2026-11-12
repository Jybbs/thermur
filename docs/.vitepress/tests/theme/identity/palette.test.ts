// @vitest-environment happy-dom
import { type DOMWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { wcagContrast }                              from 'culori'
import { afterEach, describe, expect, it, vi }       from 'vitest'
import { createSSRApp, ref }                         from 'vue'
import { renderToString }                            from 'vue/server-renderer'

import { colorOf, roles, shadeOf } from '../../../lib/identity.ts'
import source                      from '../../../lib/rendered/identity.json' with { type: 'json' }
import IdentityPalette             from '../../../theme/identity/palette.ts'
import { FLOORS, paintedIn }       from '../../common/contrast.ts'

const isDark = ref(false)

vi.mock('vitepress', () => ({ useData: () => ({ isDark }) }))

/**
 * Mounts the palette and finds the warning and critical lines of its chart.
 */
const levels = (): DOMWrapper<Element>[] => mount(IdentityPalette).findAll('.identity-level')

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

  it('draws each series’s line and spark in that series’s color', () => {
    const widget  = mount(IdentityPalette)
    const colored = source.palette.series.map((_, place) => `stroke: var(--th-series-${place + 1});`)
    expect(widget.findAll('.identity-line').map(line => line.attributes('style'))).toEqual(colored)
    expect(widget.findAll('.identity-spark path').map(spark => spark.attributes('style'))).toEqual(colored)
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
    expect(keys.map(key => key.get('i').attributes('style')))
      .toEqual(roles.map(role => `background: var(--th-${role});`))
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

describe('the warning and critical lines of IdentityPalette', () => {
  it('names each level in ink beside a swatch of its role', () => {
    expect(levels().map(level => [level.get('text').text(), level.get('text').attributes('style')]))
      .toEqual([['warning', 'fill: var(--vp-c-text-1);'], ['critical', 'fill: var(--vp-c-text-1);']])
    expect(levels().map(level => level.get('rect').attributes('style')))
      .toEqual(['warning', 'critical'].map(role => `fill: var(--th-${role}); stroke: var(--vp-c-text-1);`))
  })

  it.for(['dark', 'light'] as const)('holds each level’s name and marks to their contrast on the %s ground', scheme => {
    const ground = shadeOf('ground', scheme)
    for (const level of levels()) {
      const name = paintedIn(level.get('text').attributes('style')!, 'fill', scheme)
      const edge = paintedIn(level.get('.identity-level-edge').attributes('style')!, 'stroke', scheme)
      const ring = paintedIn(level.get('rect').attributes('style')!, 'stroke', scheme)
      expect(wcagContrast(name, ground)).toBeGreaterThanOrEqual(FLOORS.text)
      expect(Math.min(wcagContrast(edge, ground), wcagContrast(ring, ground))).toBeGreaterThanOrEqual(FLOORS.mark)
    }
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
