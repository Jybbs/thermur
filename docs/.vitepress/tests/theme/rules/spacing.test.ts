// @vitest-environment happy-dom
import { Slider }                          from '@thermur/kit'
import { enableAutoUnmount }               from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import RulesSpacing                     from '../../../theme/rules/spacing.ts'
import { RELEASED }                     from '../../common/browser.ts'
import { type Charted, charted, press } from '../../common/rules.ts'

/**
 * The parts of the chart a case reads and moves: the slider standing over the
 * track, its thumb, the toggle playing the sweep, the spacing read above the
 * handle drawn, each rule's row, and each row's reading.
 */
interface Parts {
  player  : HTMLElement
  readout : string
  rows    : SVGElement[]
  thumb   : HTMLElement
  values  : string[]
}

/**
 * Reads the chart's parts as they stand.
 */
function parts({ host }: Charted): Parts {
  const rows = [...host.querySelectorAll<SVGElement>('g.rules-fade')]
  return {
    player  : host.querySelector<HTMLElement>('.rules-play')!,
    readout : host.querySelector('.rules-handle')!.previousElementSibling!.textContent,
    rows,
    thumb   : host.querySelector<HTMLElement>('[role=slider]')!,
    values  : rows.map(row => row.lastElementChild!.textContent)
  }
}

/**
 * Moves the slider over the track to `spacing`, as a drag along it does, and
 * lets it go there where `released` is set.
 */
async function drag(chart: Charted, spacing: number, released = false): Promise<void> {
  await chart.widget.vm.$nextTick()
  const slider = chart.widget.getComponent(Slider)
  slider.vm.$emit('change', spacing)
  if (released) slider.vm.$emit('commit', spacing)
  await chart.widget.vm.$nextTick()
}

/**
 * Presses the toggle playing the sweep, once the chart has laid it out.
 */
async function play(chart: Charted): Promise<void> {
  await chart.widget.vm.$nextTick()
  parts(chart).player.click()
  await chart.widget.vm.$nextTick()
}

enableAutoUnmount(afterEach)

afterEach(() => {
  delete document.documentElement.dataset.paused
})

describe('the coupling chart', () => {
  it('draws a row per rule at the densest launch, each read as its mean beside its standard error', () => {
    const chart = charted(RulesSpacing, 672)
    const read  = parts(chart)
    expect(read.rows.map(row => row.querySelector('text')!.textContent)).toEqual([
      'Constant, the default', 'Anisotropic', 'Distance-weighted', 'Metric'
    ])
    expect(read.values).toEqual(['0.93 ± 0.01', '0.91 ± 0.02', '0.77 ± 0.06', '0.66 ± 0.07'])
    expect(read.readout).toBe('10 m')
    expect(chart.host.querySelector('svg')!.getAttribute('viewBox')).toBe('0 0 672 236')
  })

  it('sets its caption above the track on a narrow column', () => {
    const chart   = charted(RulesSpacing, 672)
    const caption = (): Element => [...chart.host.querySelectorAll('text')].find(text =>
      text.textContent === 'LAUNCH SPACING')!
    expect(caption().getAttribute('y')).toBe('28')
    chart.widen(390)
    expect(caption().getAttribute('y')).toBe('12')
    expect(chart.host.querySelector('svg')!.getAttribute('width')).toBe('390')
  })

  it('lifts the row under the pointer over the others, keeping it lifted across a redraw', () => {
    const chart   = charted(RulesSpacing, 672)
    const opacity = (): string[] => parts(chart).rows.map(row => row.style.getPropertyValue('opacity'))
    parts(chart).rows[2]!.dispatchEvent(new Event('pointerenter'))
    expect(opacity()).toEqual(['0.35', '0.35', '1', '0.35'])
    chart.widen(640)
    expect(opacity()).toEqual(['0.35', '0.35', '1', '0.35'])
    parts(chart).rows[2]!.dispatchEvent(new Event('pointerleave'))
    expect(opacity()).toEqual(['1', '1', '1', '1'])
  })

  it('leaves no listener, observer, or frame behind once it unmounts mid-sweep', async () => {
    const chart = charted(RulesSpacing, 672)
    await play(chart)
    chart.stand.frames.run(3)
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})

describe('the track', () => {
  it('stands a slider over the track drawn, named for the spacing and read in meters', async () => {
    const chart = charted(RulesSpacing, 672)
    await chart.widget.vm.$nextTick()
    const { thumb } = parts(chart)
    expect(thumb.getAttribute('aria-label')).toBe('Launch spacing')
    expect(thumb.getAttribute('aria-valuetext')).toBe('10 m')
    expect(chart.host.querySelector<HTMLElement>('.kit-slider')!.style.width).toMatch(/^[\d.]+px$/u)
  })

  it.for([
    { key: 'ArrowRight', to: '15 m' },
    { key: 'ArrowUp', to: '15 m' },
    { key: 'End', to: '25 m' },
    { key: 'ArrowLeft', to: '10 m' },
    { key: 'ArrowDown', to: '10 m' },
    { key: 'Home', to: '10 m' }
  ])('steps the handle to $to on $key', async ({ key, to }) => {
    const chart = charted(RulesSpacing, 672)
    await chart.widget.vm.$nextTick()
    parts(chart).thumb.focus()
    expect(press(parts(chart).thumb, key)).toBe(true)
    expect(parts(chart).readout).toBe(to)
  })

  it('leaves every other key to the browser', async () => {
    const chart = charted(RulesSpacing, 672)
    await chart.widget.vm.$nextTick()
    expect(press(parts(chart).thumb, 'Tab')).toBe(false)
  })
})

describe('a drag along the track', () => {
  it('reads a dragged handle between the measured spacings and settles it on the nearest once let go', async () => {
    const chart = charted(RulesSpacing, 672)
    await drag(chart, 17.5)
    expect(parts(chart).readout).toBe('17.5 m')
    expect(parts(chart).thumb.getAttribute('aria-valuetext')).toBe('17.5 m, between measured spacings')
    expect(parts(chart).values[0]).toMatch(/^≈ 0\.94$/u)
    await drag(chart, 17.5, true)
    expect(parts(chart).readout).toBe('15 m')
  })

  it('stops a running sweep under a drag', async () => {
    const chart = charted(RulesSpacing, 672)
    await play(chart)
    await drag(chart, 12)
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })
})

describe('the sweep', () => {
  it('sweeps the handle on from the densest launch and settles it on the nearest spacing once paused', async () => {
    const chart = charted(RulesSpacing, 672)
    await play(chart)
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('true')
    chart.stand.frames.run(12, 100)
    expect(parts(chart).readout).not.toBe('10 m')
    await play(chart)
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(['10 m', '15 m', '25 m']).toContain(parts(chart).readout)
    expect(chart.stand.held().frames).toBe(0)
  })

  it('names its toggle for what it plays', async () => {
    const chart = charted(RulesSpacing, 672)
    await chart.widget.vm.$nextTick()
    expect(parts(chart).player.getAttribute('aria-label')).toBe('Play the sweep across launch spacings')
  })

  it('draws the pause glyph on a redraw while it runs', async () => {
    const chart = charted(RulesSpacing, 672)
    await play(chart)
    chart.widen(500)
    await chart.widget.vm.$nextTick()
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('true')
    expect(chart.host.querySelector('.rules-handle')).not.toBeNull()
  })

  it('starts nothing for a reader who prefers reduced motion', async () => {
    const chart = charted(RulesSpacing, 672, true)
    await play(chart)
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })
})

describe('the sweep under the page’s pause', () => {
  it('settles once told the page’s pause came on', async () => {
    const chart = charted(RulesSpacing, 672)
    await play(chart)
    chart.stand.frames.run(4, 100)
    document.documentElement.dataset.paused = 'true'
    chart.stand.mutate()
    await chart.widget.vm.$nextTick()
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })

  it('settles on its next frame once the page’s pause is on', async () => {
    const chart = charted(RulesSpacing, 672)
    await play(chart)
    document.documentElement.dataset.paused = 'true'
    chart.stand.frames.run(1)
    await chart.widget.vm.$nextTick()
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })

  it('ignores a change of motion while no sweep runs', () => {
    const chart = charted(RulesSpacing, 672)
    document.documentElement.dataset.paused = 'true'
    chart.stand.mutate()
    expect(parts(chart).readout).toBe('10 m')
  })
})
