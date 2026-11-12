// @vitest-environment happy-dom
import { enableAutoUnmount, mount }            from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { heatRamp } from '../../../lib/identity.ts'
import source       from '../../../lib/rendered/identity.json' with { type: 'json' }
import * as sketch  from '../../../lib/specimen/sketch.ts'
import IdentityHeat from '../../../theme/identity/heat.ts'

enableAutoUnmount(afterEach)

describe('IdentityHeat', () => {
  it('fills its profile from the darkest step at the cool end to the lightest at the hot end', () => {
    const stops = mount(IdentityHeat).findAll('stop')
    expect(stops.map(stop => stop.attributes('offset'))).toEqual(['0%', '20%', '40%', '60%', '80%', '100%'])
    expect([stops[0]!.attributes('style'), stops.at(-1)!.attributes('style')])
      .toEqual(['stop-color: var(--th-heat-6);', 'stop-color: var(--th-heat-1);'])
  })

  it('fills the profile through its own gradient', () => {
    const widget = mount(IdentityHeat)
    const id     = widget.get('linearGradient').attributes('id')
    expect(widget.findAll('path')[0]!.attributes('fill')).toBe(`url(#${id})`)
  })

  it('fixes its knot at one height, reading the air there and the step it takes', () => {
    const widget   = mount(IdentityHeat)
    const { knot } = sketch.PROFILE
    const kelvin   = sketch.warmthAt(knot)
    const place    = sketch.stepAt(kelvin, heatRamp(source).length)
    const circle   = widget.get('.identity-knot')
    expect([circle.attributes('cx'), circle.attributes('cy')])
      .toEqual([String(sketch.profileX(kelvin)), String(sketch.profileY(knot))])
    expect(widget.get('.identity-readout .identity-caps').text())
      .toBe(`${knot} m · ${Math.round(kelvin)} K · Step ${place + 1}`)
  })

  it('copies the hex of the knot’s step from its chip', async () => {
    const widget = mount(IdentityHeat, { attachTo: document.body })
    const ramp   = heatRamp(source)
    const hex    = ramp[sketch.stepAt(sketch.warmthAt(sketch.PROFILE.knot), ramp.length)]!
    const chip   = widget.get('.identity-chip')
    expect(chip.get('.identity-hex').text()).toBe(hex)
    await chip.trigger('click')
    await vi.waitFor(async () => expect(await navigator.clipboard.readText()).toBe(hex))
    await vi.waitFor(() => expect(widget.get('[aria-live]').text()).toBe(`Copied ${hex}`))
  })
})
