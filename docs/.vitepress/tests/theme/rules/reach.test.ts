// @vitest-environment happy-dom
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import RulesReach                       from '../../../theme/rules/reach.ts'
import { RELEASED }                       from '../../common/browser.ts'
import { type Charted, charted, press } from '../../common/rules.ts'

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
 * Finds the stops on the dial's rim, one per turn.
 */
const knobs = ({ host }: Charted): SVGElement[] => [...host.querySelectorAll<SVGElement>('[role="radio"]')]

enableAutoUnmount(afterEach)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

describe('the heading chart', () => {
  it('reads each rule’s reach at the smallest turn, named in full beside a wide dial', () => {
    const chart = charted(RulesReach, 672)
    expect(words(chart)).toEqual(expect.arrayContaining(['Inertial spin, the default', 'Consensus', 'Delayed', 'Nonmutual', 'Quartic']))
    expect(readings(chart)).toEqual([
      '76.6°', 'of 20° · 383% ± 118',
      '99.9°', 'of 20° · 500% ± 75',
      '7.3°', 'of 20° · 37% ± 5',
      '46.7°', 'of 20° · 233% ± 70',
      '117.9°', 'of 20° · 589% ± 59'
    ])
    expect(knobs(chart).map(knob => knob.getAttribute('aria-checked'))).toEqual(['true', 'false'])
  })

  it('sets the shorter names under the dial on a narrow column', () => {
    const chart = charted(RulesReach, 672)
    chart.widen(390)
    expect(words(chart)).toEqual(expect.arrayContaining(['Inertial spin', 'Consensus', 'Delayed', 'Nonmutual', 'Quartic']))
    expect(chart.host.querySelector('svg')!.getAttribute('width')).toBe('390')
  })

})

describe('the stops on the dial’s rim', () => {
  it('sweeps to the turn picked on the rim, reading each rule’s reach of it', () => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[1]!.dispatchEvent(new Event('click'))
    expect(readings(chart)).toEqual([
      '95.6°', 'of 90° · 106% ± 23',
      '75.2°', 'of 90° · 84% ± 12',
      '7.2°', 'of 90° · 8% ± 2',
      '55.5°', 'of 90° · 62% ± 17',
      '97.3°', 'of 90° · 108% ± 17'
    ])
    expect(knobs(chart).map(knob => knob.getAttribute('tabindex'))).toEqual(['-1', '0'])
  })

  it.for([
    { from: 0, key: 'ArrowRight', to: 1 },
    { from: 0, key: 'ArrowUp', to: 1 },
    { from: 0, key: 'ArrowLeft', to: 1 },
    { from: 1, key: 'ArrowDown', to: 0 },
    { from: 1, key: 'Enter', to: 1 },
    { from: 1, key: ' ', to: 1 }
  ])('moves the pick from stop $from to stop $to on $key, focus and all', ({ from, key, to }) => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[from]!.dispatchEvent(new Event('click'))
    expect(press(knobs(chart)[from]!, key)).toBe(true)
    expect(knobs(chart)[to]!.getAttribute('aria-checked')).toBe('true')
    expect(document.activeElement).toBe(knobs(chart)[to])
  })

  it('leaves every other key to the browser', () => {
    const chart = charted(RulesReach, 672)
    expect(press(knobs(chart)[0]!, 'Tab')).toBe(false)
  })

})

describe('the rings', () => {
  it('lifts the rule under the pointer over the others', () => {
    const chart   = charted(RulesReach, 672)
    const groups  = [...chart.host.querySelectorAll<SVGElement>('g.rules-fade')]
    const opacity = (): string[] => groups.map(group => group.style.getPropertyValue('opacity'))
    groups[2]!.dispatchEvent(new Event('pointerenter'))
    expect(opacity()).toEqual(['0.3', '0.3', '1', '1', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3'])
    groups[3]!.dispatchEvent(new Event('pointerleave'))
    expect(new Set(opacity())).toEqual(new Set(['1']))
  })

  it.for([0, 1])('stops each ring at the first drone’s turn on stop %i, inside the dial’s quarter', stop => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[stop]!.dispatchEvent(new Event('click'))
    const paths  = [...chart.host.querySelectorAll('path')].map(path => path.getAttribute('d')!)
    const center = Number(paths.find(d => d.includes(' L'))!.slice(1).split(',')[0])
    const xs     = [...paths.join(' ').matchAll(/(-?[\d.]+),-?[\d.]+/gu)].map(([, x]) => Number(x))
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(center - 1e-6)
  })

  it('staggers the rings over the theme’s settling time, outer first', () => {
    document.documentElement.style.setProperty('--th-settle', '240ms')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    const chart = charted(RulesReach, 672)
    chart.stand.frames.run(1, 400)
    const [outer, , inner] = readings(chart).filter(line => line.endsWith('°'))
    expect(Number(outer!.slice(0, -1)) / 76.6).toBeGreaterThan(Number(inner!.slice(0, -1)) / 7.3)
    chart.stand.frames.run(20, 400)
    expect(readings(chart)[0]).toBe('76.6°')
  })

  it('leaves no listener, observer, or frame behind once it unmounts mid-sweep', () => {
    document.documentElement.style.setProperty('--th-settle', '240ms')
    const chart = charted(RulesReach, 672)
    knobs(chart)[1]!.dispatchEvent(new Event('click'))
    chart.stand.frames.run(2)
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})
