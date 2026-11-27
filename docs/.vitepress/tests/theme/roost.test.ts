// @vitest-environment happy-dom
import { type VueWrapper, enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PLAN }                             from '../../lib/roost/flight.ts'
import RoostView                            from '../../theme/roost.ts'
import { type Browser, RELEASED, browser }  from '../common/browser.ts'

/**
 * A mounted view over the stand-in browser, beside the stand-ins, once its
 * flight is ready.
 */
async function mounted(reduced = false): Promise<{ stand: Browser, view: VueWrapper }> {
  const stand = browser(reduced)
  const view  = mount(RoostView, { attachTo: document.body })
  await vi.waitFor(() => expect(view.find('canvas').exists()).toBe(true), { timeout: 30_000 })
  stand.see(true)
  return { stand, view }
}

/**
 * Reads the chips over the scene, each its name and its value.
 */
const chips = (view: VueWrapper): string[] => view.findAll('.kit-chip').map(chip => chip.text())

enableAutoUnmount(afterEach)

describe('the roost view', () => {
  it('sets its scene in an image naming what it shows, over three readings and the note', async () => {
    const { view } = await mounted()
    expect(view.get('[role=img]').attributes('aria-label')).toContain('Two hundred drones')
    expect(view.findAll('.roost-reading').map(reading => reading.find('.roost-name').text())).toEqual(['Polarization', 'Spacing', 'Height'])
    expect(view.get('.roost-note').text()).toContain('200 drones')
  })

  it('writes the clock, the ringed height, and each reading as the flight plays', async () => {
    const { stand, view } = await mounted()
    stand.frames.run(40, 100)
    expect(chips(view)[0]).toMatch(/^Time\d+ s$/u)
    expect(chips(view)[1]).toMatch(/^Ringed height\d+ m$/u)
    expect(view.findAll('.roost-value').map(value => value.text()).every(Boolean)).toBe(true)
  })

  it('shows the still moment once the flight is flown to it, while motion stands still', async () => {
    const { view } = await mounted(true)
    expect(chips(view)[0]).toBe(`Time${PLAN.still} s`)
  })

  it('leaves no listener, observer, animation frame, or canvas behind once it unmounts', async () => {
    const { stand, view } = await mounted()
    stand.frames.run(3)
    expect(stand.held().canvases).toBe(4)
    view.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('draws nothing where it unmounts before its flight is flown', async () => {
    const stand = browser(true)
    const view  = mount(RoostView, { attachTo: document.body })
    view.unmount()
    await flushPromises()
    expect(stand.held().canvases).toBe(0)
  })
})
