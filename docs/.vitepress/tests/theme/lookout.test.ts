// @vitest-environment happy-dom
import { Slider }                                            from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, flushPromises } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }               from 'vitest'

import { DRONES }                         from '../../lib/lookout/flocks.ts'
import LookoutFlight                      from '../../theme/lookout.ts'
import { type Browser, RELEASED }         from '../common/browser.ts'
import { charted }                        from '../common/rules.ts'

/**
 * Mounts the Lookout 672 pixels across, its frames running, once its first
 * flight is in place.
 */
async function mounted(): Promise<{ stand: Browser, widget: VueWrapper }> {
  const { stand, widget } = charted(LookoutFlight, 672)
  await flushPromises()
  stand.see(true)
  stand.frames.run(2, 50)
  return { stand, widget }
}

/**
 * Reads the chips beside the flight, each its name and its value.
 */
const chips = (widget: VueWrapper): string[] => widget.findAll('.kit-chip').map(chip => chip.text())

/**
 * Lets go of the slider at `count` drones and waits for the flight it flies.
 */
async function commit(widget: VueWrapper, count: number): Promise<void> {
  widget.getComponent(Slider).vm.$emit('commit', count)
  await flushPromises()
}

/**
 * Answers as `thermur watch` does with a flight of `count` drones two frames
 * long.
 */
const served = (count: number): Response => Response.json({
  drones     : count,
  frame_s    : 0.5,
  positions  : Array.from({ length: count * 6 }, (_, at) => at),
  velocities : Array.from({ length: count * 6 }, () => 1)
})

enableAutoUnmount(afterEach)

afterEach(() => {
  window.history.replaceState(null, '', '/')
  vi.unstubAllGlobals()
})

describe('the Lookout', () => {
  it('flies the plan’s flock in the page, its readings beside it', async () => {
    const { widget } = await mounted()
    const [drones, time, polarization, groups] = chips(widget)
    expect(drones).toBe(`Drones${DRONES.value.toLocaleString('en-US')}`)
    expect(time).toMatch(/^Time\d+\.\d s$/u)
    expect(polarization).toMatch(/^Polarization[01]\.\d\d$/u)
    expect(groups).toMatch(/^Groups\d+$/u)
  })

  it('names its slider for the flock’s size, marking the default on the track', async () => {
    const { widget } = await mounted()
    expect(widget.get('[role=slider]').attributes()).toMatchObject({
      'aria-label'     : 'Drones in the flock',
      'aria-valuemax'  : String(DRONES.max),
      'aria-valuemin'  : String(DRONES.min),
      'aria-valuetext' : `${DRONES.value.toLocaleString('en-US')} drones, the default`
    })
    const share = 100 * (DRONES.value - DRONES.min) / (DRONES.max - DRONES.min)
    expect(widget.get('.lookout-default').attributes('style')).toContain(`left: ${share}%`)
  })

  it('flies a size let go from its start', async () => {
    const { stand, widget } = await mounted()
    widget.getComponent(Slider).vm.$emit('change', 40)
    await widget.vm.$nextTick()
    expect(widget.get('[role=slider]').attributes('aria-valuenow')).toBe('40')
    await commit(widget, 40)
    expect(chips(widget)[0]).toBe('Drones40')
    expect(chips(widget)[1]).toBe('Time0.0 s')
    stand.frames.run(10, 50)
    expect(chips(widget)[1]).not.toBe('Time0.0 s')
  })

})

describe('the Lookout under `thermur watch`', () => {
  it('plays the flight `thermur watch` serves where the page was opened on one, the size let go last winning', async () => {
    window.history.replaceState(null, '', '/methods/lookout?flight=/flight')
    const answers = new Map<string, (answer: Response) => void>()
    vi.stubGlobal('fetch', (address: string) => new Promise<Response>(resolve => {
      answers.set(address, resolve)
    }))
    const { widget } = charted(LookoutFlight, 672)
    answers.get(`/flight?drones=${DRONES.value}`)!(served(DRONES.value))
    await flushPromises()
    expect(chips(widget)[0]).toBe(`Drones${DRONES.value.toLocaleString('en-US')}`)
    widget.getComponent(Slider).vm.$emit('commit', 30)
    widget.getComponent(Slider).vm.$emit('commit', 20)
    await flushPromises()
    answers.get('/flight?drones=20')!(served(20))
    await flushPromises()
    answers.get('/flight?drones=30')!(served(30))
    await flushPromises()
    expect(chips(widget)[0]).toBe('Drones20')
  })

})

describe('the Lookout’s teardown', () => {
  it('leaves no listener, observer, animation frame, or canvas behind once it unmounts', async () => {
    const { stand, widget } = await mounted()
    stand.frames.run(3)
    expect(stand.held().canvases).toBe(1)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('draws nothing where it unmounts before its first flight is in place', async () => {
    const { stand, widget } = charted(LookoutFlight, 672)
    widget.unmount()
    await flushPromises()
    expect(stand.held()).toEqual(RELEASED)
  })
})
