// @vitest-environment happy-dom
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import RulesReach                from '../../../theme/rules/reach.ts'
import { RELEASED }              from '../../common/browser.ts'
import { type Charted, charted } from '../../common/rules.ts'

/**
 * Reads the words of every line of text in the chart.
 */
const words = ({ host }: Charted): string[] =>
  [...host.querySelectorAll('text')].map(text => text.textContent)

/**
 * Reads each rule's angle beside the share of the turn it reaches.
 */
const readings = (chart: Charted): string[] => words(chart).filter(line => /^\d+\.\d°$|^of /u.test(line))

/**
 * Finds the radios standing over the stops on the dial's rim, one per turn.
 */
const knobs = ({ host }: Charted): HTMLInputElement[] => [...host.querySelectorAll<HTMLInputElement>('input[type=radio]')]

/**
 * Reads which stops drawn on the rim show their dot.
 */
const dots = ({ host }: Charted): (string | undefined)[] =>
  [...host.querySelectorAll<SVGElement>('.rules-knob')].map(knob => knob.dataset.on)

enableAutoUnmount(afterEach)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

describe('the heading chart', () => {
  it('reads each rule’s reach at the smallest turn, named in full beside a wide dial', () => {
    const chart = charted(RulesReach, 672)
    expect(words(chart)).toEqual(expect.arrayContaining(['Overdamped, the default', 'Consensus', 'Delayed']))
    expect(readings(chart)).toEqual([
      '16.4°', 'of 20° · 82% ± 18',
      '15.7°', 'of 20° · 78% ± 20',
      '6.7°', 'of 20° · 34% ± 12'
    ])
    expect(knobs(chart).map(knob => knob.checked)).toEqual([true, false])
    expect(dots(chart)).toEqual(['true', 'false'])
  })

  it('sets the shorter names under the dial on a narrow column', () => {
    const chart = charted(RulesReach, 672)
    chart.widen(390)
    expect(words(chart)).toEqual(expect.arrayContaining(['Overdamped', 'Consensus', 'Delayed']))
    expect(chart.host.querySelector('svg')!.getAttribute('width')).toBe('390')
  })

})

describe('the stops on the dial’s rim', () => {
  it('sweeps to the turn picked on the rim, reading each rule’s reach of it', async () => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[1]!.click()
    await chart.widget.vm.$nextTick()
    expect(readings(chart)).toEqual([
      '19.2°', 'of 90° · 21% ± 4',
      '20.4°', 'of 90° · 23% ± 7',
      '8.8°', 'of 90° · 10% ± 4'
    ])
    expect(knobs(chart).map(knob => knob.checked)).toEqual([false, true])
    expect(dots(chart)).toEqual(['false', 'true'])
  })

  it('names the group of radios and each radio for its turn, one name shared so the arrow keys move between them', () => {
    const chart = charted(RulesReach, 672)
    const group = chart.host.querySelector('[role=radiogroup]')!
    expect(document.querySelector(`[id="${group.getAttribute('aria-labelledby')!}"]`)?.textContent).toBe('The first drone turns')
    expect([...chart.host.querySelectorAll('.kit-radios-item')].map(item => item.textContent)).toEqual(['20°', '90°'])
    expect(new Set(knobs(chart).map(knob => knob.name)).size).toBe(1)
  })

  it('stands each radio over its stop on the rim, again at each width', async () => {
    const chart = charted(RulesReach, 672)
    await chart.widget.vm.$nextTick()
    const places = (): string[] => [...chart.host.querySelectorAll<HTMLElement>('.kit-radios-item')]
      .map(item => `${item.style.left} ${item.style.top}`)
    const wide   = places()
    chart.widen(390)
    await chart.widget.vm.$nextTick()
    expect(wide.every(place => /^[\d.]+px [\d.]+px$/u.test(place))).toBe(true)
    expect(places()).not.toEqual(wide)
  })
})

describe('the rings', () => {
  it('lifts the rule under the pointer over the others', () => {
    const chart   = charted(RulesReach, 672)
    const groups  = [...chart.host.querySelectorAll<SVGElement>('g.rules-fade')]
    const opacity = (): string[] => groups.map(group => group.style.getPropertyValue('opacity'))
    groups[2]!.dispatchEvent(new Event('pointerenter'))
    expect(opacity()).toEqual(['0.3', '0.3', '1', '1', '0.3', '0.3'])
    groups[3]!.dispatchEvent(new Event('pointerleave'))
    expect(new Set(opacity())).toEqual(new Set(['1']))
  })

  it('staggers the rings over the theme’s settling time, outer first', () => {
    document.documentElement.style.setProperty('--th-settle', '240ms')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    const chart = charted(RulesReach, 672)
    chart.stand.frames.run(1, 400)
    const [outer, , inner] = readings(chart).filter(line => line.endsWith('°'))
    expect(Number(outer!.slice(0, -1)) / 16.4).toBeGreaterThan(Number(inner!.slice(0, -1)) / 6.7)
    chart.stand.frames.run(20, 400)
    expect(readings(chart)[0]).toBe('16.4°')
  })

  it('leaves no listener, observer, or frame behind once it unmounts mid-sweep', () => {
    document.documentElement.style.setProperty('--th-settle', '240ms')
    const chart = charted(RulesReach, 672)
    knobs(chart)[1]!.click()
    chart.stand.frames.run(2)
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})
