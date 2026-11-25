// @vitest-environment happy-dom
import { idle }                                from '@thermur/kit'
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h }                  from 'vue'

import { useStage }              from '../../../../docs/.vitepress/lib/canvas/mount.ts'
import type { Rule }             from '../../../../docs/.vitepress/lib/rules/measured.ts'
import { Reach }                 from '../../../../docs/.vitepress/lib/rules/reach.ts'
import RulesReach                from '../../../../docs/.vitepress/theme/rules/reach.ts'
import { RELEASED }              from '../../common/browser.ts'
import { type Charted, charted } from '../../common/rules.ts'

/**
 * Two rules on the 20° and 90° turns, the second's farthest drones turning
 * past the first drone on both, out to 44° and 144° with their errors.
 */
const PAST: readonly Rule[] = [
  {
    color  : '#bb7733',
    label  : 'Within',
    points : [{ error: 0.1, given: 20, mean: 0.5 }, { error: 0.1, given: 90, mean: 0.5 }],
    short  : 'Within'
  },
  {
    color  : '#5069ad',
    label  : 'Past',
    points : [{ error: 0.2, given: 20, mean: 2 }, { error: 0.1, given: 90, mean: 1.5 }],
    short  : 'Past'
  }
]
/**
 * The heading chart drawn for `PAST`.
 */
const Past = defineComponent({
  name: 'RulesReachPast',
  setup() {
    const host = useStage(box => new Reach(box, idle, PAST))
    return () => h('figure', [h('div', { ref: host })])
  }
})

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

/**
 * Measures how far the end of `node`'s words falls from its start, at the
 * advance of 0.6 of its size that every glyph of the site's monospace face
 * takes.
 */
const across = (node: Element): number => node.textContent.length * 0.6 * Number(node.getAttribute('font-size'))

/**
 * Finds where `node`'s words start, from its `x` and its anchor.
 */
const startOf = (node: Element): number =>
  Number(node.getAttribute('x')) - (node.getAttribute('text-anchor') === 'end' ? across(node) : 0)

/**
 * Measures the room between the end of `angle` and the start of the nearest
 * words after it on its line.
 */
const roomAfter = (angle: Element, texts: readonly Element[]): number => Math.min(...texts
  .filter(text => text.getAttribute('y') === angle.getAttribute('y') && startOf(text) > startOf(angle))
  .map(text => startOf(text))) - startOf(angle) - across(angle)

/**
 * Reads the widths of the track and the bar drawn in `color`.
 */
const barsOf = ({ host }: Charted, color: string): number[] => [...host.querySelectorAll('rect')]
  .filter(rect => rect.getAttribute('fill') === color && rect.getAttribute('rx') === '2')
  .map(rect => Number(rect.getAttribute('width')))

/**
 * Reads the path of the full ring and of the lighter ring past the first
 * drone's turn, for each rule.
 */
const arcs = ({ host }: Charted): [string, string][] =>
  [...host.querySelectorAll('g.rules-fade')]
    .filter(group => group.querySelector('path'))
    .map(group => {
      const paths = [...group.querySelectorAll('path')].map(path => path.getAttribute('d') ?? '')
      return [paths[1]!, paths[3]!]
    })

/**
 * Finds the x of the dial's center, where the marker of the first drone's
 * turn starts.
 */
const centerOf = ({ host }: Charted): number =>
  Number([...host.querySelectorAll('path')].map(path => path.getAttribute('d')!).find(d => d.includes(' L'))!.slice(1).split(',')[0])

/**
 * Reads the x of every point the chart's paths pass through.
 */
const xsOf = ({ host }: Charted): number[] =>
  [...[...host.querySelectorAll('path')].map(path => path.getAttribute('d') ?? '').join(' ').matchAll(/(-?[\d.]+),-?[\d.]+/gu)]
    .map(([, x]) => Number(x))

enableAutoUnmount(afterEach)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

describe('the heading chart', () => {
  it('reads each rule’s reach at the smallest turn, named in full beside a wide dial', () => {
    const chart = charted(RulesReach, 1024)
    expect(words(chart)).toEqual(expect.arrayContaining(['Consensus, the default', 'Inertial spin', 'Delayed', 'Nonmutual', 'Quartic']))
    expect(readings(chart)).toEqual([
      '0.1°', 'of 20° · 0.3% ± 0.3',
      '0.5°', 'of 20° · 2.6% ± 2.2',
      '0.1°', 'of 20° · 0.3% ± 0.2',
      '0.5°', 'of 20° · 2.5% ± 2.0',
      '0.6°', 'of 20° · 3.2% ± 2.8'
    ])
    expect(knobs(chart).map(knob => knob.checked)).toEqual([true, false])
    expect(dots(chart)).toEqual(['true', 'false'])
  })

  it.for([1024, 672, 390])('leaves a clear gap between each angle and the next words on its line at %i px', width => {
    const chart  = charted(RulesReach, width)
    const texts  = [...chart.host.querySelectorAll('text')]
    const angles = texts.filter(text => /^\d+\.\d°$/u.test(text.textContent))
    expect(angles).toHaveLength(5)
    expect(angles.map(angle => roomAfter(angle, texts)).filter(room => room < 7)).toStrictEqual([])
  })

  it('sets the shorter names beside each angle between the narrow and the wide widths', () => {
    const chart = charted(RulesReach, 672)
    expect(words(chart)).toEqual(expect.arrayContaining(['Consensus', 'Inertial spin', 'Delayed', 'Nonmutual', 'Quartic']))
    expect(words(chart)).not.toContain('Consensus, the default')
  })

  it('sets the shorter names under the dial on a narrow column', () => {
    const chart = charted(RulesReach, 672)
    chart.widen(390)
    expect(words(chart)).toEqual(expect.arrayContaining(['Consensus', 'Inertial spin', 'Delayed', 'Nonmutual', 'Quartic']))
    expect(chart.host.querySelector('svg')!.getAttribute('width')).toBe('390')
  })

})

describe('the stops on the dial’s rim', () => {
  it('sweeps to the turn picked on the rim, reading each rule’s reach of it', async () => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[1]!.click()
    await chart.widget.vm.$nextTick()
    expect(readings(chart)).toEqual([
      '0.2°', 'of 90° · 0.2% ± 0.2',
      '0.5°', 'of 90° · 0.6% ± 0.3',
      '0.2°', 'of 90° · 0.2% ± 0.2',
      '0.3°', 'of 90° · 0.3% ± 0.2',
      '2.1°', 'of 90° · 2.3% ± 1.9'
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

  it('sets the caption over the radios as every control’s label is set', () => {
    const chart   = charted(RulesReach, 672)
    const caption = [...chart.host.querySelectorAll('text')]
      .find(text => text.textContent === 'THE FIRST DRONE TURNS')!
    expect(caption).toSetAsControlLabel()
  })

  it('stands each radio over its stop on the rim, again at each width', async () => {
    const chart = charted(RulesReach, 672)
    await chart.widget.vm.$nextTick()
    const places = (): string[] => [...chart.host.querySelectorAll<HTMLElement>('.kit-radios-item')]
      .map(item => `${item.style.left} ${item.style.top}`)
    const wide   = places()
    chart.widen(390)
    await chart.widget.vm.$nextTick()
    expect(wide.filter(place => !/^[\d.]+px [\d.]+px$/u.test(place))).toStrictEqual([])
    expect(places()).not.toEqual(wide)
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

  it.for([0, 1])('draws every measured ring inside the dial’s quarter on stop %i, none turning past the first drone', stop => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[stop]!.dispatchEvent(new Event('click'))
    expect(Math.min(...xsOf(chart))).toBeGreaterThanOrEqual(centerOf(chart) - 1e-6)
    expect(arcs(chart).map(([, past]) => past)).toEqual(['', '', '', '', ''])
  })

  it.for([0, 1])('holds each bar’s whisker inside the chart on stop %i', stop => {
    const chart = charted(RulesReach, 672)
    knobs(chart)[stop]!.dispatchEvent(new Event('click'))
    const width = Number(chart.host.querySelector('svg')!.getAttribute('width'))
    const ends  = [...chart.host.querySelectorAll('line')]
      .flatMap(line => [line.getAttribute('x1'), line.getAttribute('x2')])
      .map(Number)
    expect(Math.max(...ends)).toBeLessThanOrEqual(width)
  })

  it('staggers the rings over the theme’s settling time, outer first', () => {
    document.documentElement.style.setProperty('--th-settle', '240ms')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    const chart = charted(Past, 672)
    chart.stand.frames.run(1, 300)
    const [outer, inner] = readings(chart).filter(line => line.endsWith('°'))
    expect(Number(outer!.slice(0, -1)) / 10).toBeGreaterThan(Number(inner!.slice(0, -1)) / 40)
    chart.stand.frames.run(20, 400)
    expect(readings(chart).filter(line => line.endsWith('°'))).toEqual(['10.0°', '40.0°'])
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

describe('the rings past the first drone’s turn', () => {
  it('draws a lighter ring past the first drone’s turn, on a frame running past the vertical', () => {
    const chart          = charted(Past, 672)
    const [within, past] = arcs(chart)
    expect(within).toEqual([expect.stringMatching(/^M/u), ''])
    expect(past![1]).toMatch(/^M/u)
    expect(Math.min(...xsOf(chart))).toBeLessThan(centerOf(chart))
    const [whole, bar] = barsOf(chart, '#5069ad')
    expect(bar).toBe(whole)
  })
})

describe('take A of the heading chart', () => {
  it('draws the lighter band, the frame, and a disc under each stop past the first drone’s turn', () => {
    const chart = charted(Past, 672)
    const svg   = chart.host.querySelector('svg')!
    const own   = (name: string): Element[] => [...svg.children].filter(node => node.localName === name)
    const past  = [...chart.host.querySelectorAll('g.rules-fade')].findLast(group => group.querySelector('path'))!
    const [, , , pastArc, pastBand] = [...past.querySelectorAll('path')]
    expect(pastBand!.getAttribute('d')).toMatch(/^M/u)
    expect(Number((pastArc as SVGElement).style.opacity)).toBe(0.4)
    expect(Number(own('path')[0]!.getAttribute('d')!.split(' ').at(-1)!.split(',')[0])).toBeLessThan(centerOf(chart))
    expect(own('line')).toHaveLength(6)
    expect(own('circle')).toHaveLength(2)
  })

  it.for([[1024, 56], [672, 40]] as const)('sets the readings in a column as wide as the dial at %i px', ([width, gutter]) => {
    const chart  = charted(RulesReach, width)
    const tracks = [...chart.host.querySelectorAll('rect')].filter(rect => rect.style.opacity === '0.15')
    expect(tracks).toHaveLength(5)
    expect(tracks.map(track => Number(track.getAttribute('width'))))
      .toStrictEqual(tracks.map(track => Number(track.getAttribute('x')) - gutter))
  })

  it('fills half the chart with the dial where its radius stays under the cap', () => {
    const chart  = charted(RulesReach, 672)
    const tracks = [...chart.host.querySelectorAll('rect')].filter(rect => rect.style.opacity === '0.15')
    expect(tracks).toHaveLength(5)
    const ends = tracks.map(track => Number(track.getAttribute('x')) + Number(track.getAttribute('width')))
    expect(ends.filter(end => end <= 672 - 4)).toStrictEqual([])
  })
})
