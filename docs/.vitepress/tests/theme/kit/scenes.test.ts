// @vitest-environment happy-dom
import { Radios, Select, Slider, Toggle }            from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }       from 'vitest'

import { FUELS, GROUND, SPAN, reachOf } from '../../../lib/kit/front.ts'
import { SCENES }                       from '../../../theme/kit/scenes.ts'
import { RELEASED, browser }            from '../../common/browser.ts'

enableAutoUnmount(afterEach)

afterEach(() => {
  delete document.documentElement.dataset.paused
  document.documentElement.style.removeProperty('--th-settle')
})

/**
 * Mounts the scene of `part` into the document over the stand-in browser,
 * standing still where `reduced` holds.
 */
function staged(part: string, reduced = false): { stand: ReturnType<typeof browser>, widget: VueWrapper } {
  const stand  = browser(reduced)
  const widget = mount(SCENES[part]!, { attachTo: document.body })
  return { stand, widget }
}

/**
 * Reads the width the front's burned ground is drawn at.
 */
const burned = (widget: VueWrapper): number => Number(widget.get('.kit-catalog-front rect').attributes('width'))

describe('the controls scene', () => {
  it('moves one front as the fuel, the wind, and the seconds change', async () => {
    const { widget } = staged('controls')
    const scale      = 568 / GROUND
    expect(burned(widget)).toBeCloseTo(reachOf(0, 1, 45) * scale)
    widget.getComponent(Select).vm.$emit('change', 2)
    widget.getComponent(Radios).vm.$emit('change', 'gale')
    widget.getComponent(Slider).vm.$emit('change', 30)
    await widget.vm.$nextTick()
    expect(burned(widget)).toBeCloseTo(FUELS[2].rate * 2 * 30 * scale)
    widget.getComponent(Slider).vm.$emit('commit', 60)
    await widget.vm.$nextTick()
    expect(burned(widget)).toBeCloseTo(reachOf(2, 2, 60) * scale)
  })

  it('plays the front to the end of its run and stops there, starting over from ignition', async () => {
    const { stand, widget } = staged('controls')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    widget.getComponent(Toggle).vm.$emit('change', true)
    stand.frames.run(20, 500)
    await widget.vm.$nextTick()
    expect(widget.getComponent(Slider).props('value')).toBe(SPAN)
    expect(widget.getComponent(Toggle).props('pressed')).toBe(false)
    widget.getComponent(Toggle).vm.$emit('change', true)
    await widget.vm.$nextTick()
    expect(widget.getComponent(Slider).props('value')).toBe(0)
  })

  it('jumps to the end of the run for a reader who prefers reduced motion', async () => {
    const { stand, widget } = staged('controls', true)
    widget.getComponent(Toggle).vm.$emit('change', true)
    await widget.vm.$nextTick()
    expect(widget.getComponent(Slider).props('value')).toBe(SPAN)
    expect(stand.held().frames).toBe(0)
  })

  it('lays its front out at the width the plate stands at', () => {
    const { stand, widget } = staged('controls')
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({ width: 320 } as DOMRect)
    stand.reflow()
    expect(widget.get('.kit-catalog-front').attributes('viewBox')).toBe('0 0 320 112')
  })

  it('releases every frame and listener once it unmounts mid-play', () => {
    const { stand, widget } = staged('controls')
    widget.getComponent(Toggle).vm.$emit('change', true)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})

describe('the chrome scene', () => {
  it('reads the time and the reach on its chips, stepping a snapshot on and wrapping past the end', async () => {
    const { widget } = staged('chrome')
    const values     = (): string[] => widget.findAll('.kit-chip').map(chip => chip.text())
    expect(values()).toEqual([expect.stringContaining('30 s'), expect.stringContaining(`${reachOf(0, 1, 30)} m`)])
    const next = widget.findAll('button').find(button => button.text() === 'Next snapshot')!
    for (let press = 0; press < 6; press++) (next.element as HTMLElement).click()
    await widget.vm.$nextTick()
    expect(values()[0]).toContain('120 s')
    await next.trigger('click')
    expect(values()[0]).toContain('0 s')
  })

  it('copies a step of the heat ramp from its button', async () => {
    const { widget } = staged('chrome')
    const step       = widget.get('.kit-catalog-step')
    await step.trigger('click')
    await vi.waitFor(async () => expect(await navigator.clipboard.readText()).toBe(step.get('.kit-fuse').text()))
  })
})

describe('the motion scene', () => {
  it('flies the flock while its toggle is pressed', async () => {
    const { stand, widget } = staged('motion')
    const bird              = (): string | undefined => widget.get('.kit-catalog-flight circle').attributes('cx')
    const before            = bird()
    widget.findAllComponents(Toggle)[0]!.vm.$emit('change', true)
    stand.frames.run(4, 100)
    await widget.vm.$nextTick()
    expect(bird()).not.toBe(before)
  })

  it('turns the page’s pause on and off, saying whether motion runs', async () => {
    const { stand, widget } = staged('motion')
    const gate              = widget.findAllComponents(Toggle)[1]!
    expect(widget.text()).toContain('Motion runs')
    gate.vm.$emit('change', true)
    stand.mutate('data-paused')
    await widget.vm.$nextTick()
    expect([document.documentElement.dataset.paused, gate.text()]).toEqual(['true', 'Resume'])
    expect(widget.text()).toContain('Motion stands still')
  })

  it('gathers the flock in and lets it loose again over the settling time', async () => {
    const { stand, widget } = staged('motion')
    document.documentElement.style.setProperty('--th-settle', '100ms')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    const spread = (): number => {
      const xs = widget.findAll('.kit-catalog-flight circle').map(circle => Number(circle.attributes('cx')))
      return Math.max(...xs) - Math.min(...xs)
    }
    const loose  = spread()
    const gather = (): ReturnType<VueWrapper['get']> => widget.get('[data-piece="timing"] button')
    await gather().trigger('click')
    expect(gather().text()).toBe('Let it loose')
    stand.frames.run(30, 16)
    expect(spread()).toBeLessThan(loose / 4)
    await gather().trigger('click')
    stand.frames.run(30, 16)
    expect(spread()).toBeCloseTo(loose)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})

describe('the drawing scene', () => {
  it('draws the heat ramp at the width it stands at, again only as that width changes', () => {
    const { stand, widget } = staged('drawing')
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({ width: 480 } as DOMRect)
    stand.reflow()
    expect(widget.get('.kit-catalog-width svg').attributes('width')).toBe('480')
    expect(widget.get('.kit-catalog-width text').text()).toBe('DRAWN AT 480 PX')
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})
