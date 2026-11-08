// @vitest-environment happy-dom
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import RulesSpacing                     from '../../../theme/rules/spacing.ts'
import { RELEASED }                     from '../../common/browser.ts'
import { type Charted, charted, press } from '../../common/rules.ts'

/**
 * The parts of the chart a case reads and moves: the track's handle, the
 * strip a press lands on, the knob playing the sweep, the spacing read above
 * the handle, each rule's row, and each row's reading.
 */
interface Parts {
  handle  : SVGElement
  hit     : SVGElement
  player  : SVGElement
  readout : string
  rows    : SVGElement[]
  values  : string[]
}

/**
 * Reads the chart's parts as they stand.
 */
function parts({ host }: Charted): Parts {
  const handle = host.querySelector<SVGElement>('[role="slider"]')!
  const rows   = [...host.querySelectorAll<SVGElement>('g.rules-fade')]
  return {
    handle,
    hit     : host.querySelector<SVGElement>('.rules-track')!,
    player  : host.querySelector<SVGElement>('[role="button"]')!,
    readout : handle.previousElementSibling!.textContent,
    rows,
    values  : rows.map(row => row.lastElementChild!.textContent)
  }
}

/**
 * Lets the strip under a press hold the pointer, as a browser's pointer
 * capture does.
 */
function captured(hit: SVGElement): void {
  Object.assign(hit, { hasPointerCapture: () => true, setPointerCapture: vi.fn<(id: number) => void>() })
}

/**
 * Moves the pointer `kind` over `target` at `clientX`.
 */
const pointer = (target: Element, kind: string, clientX = 0): boolean =>
  target.dispatchEvent(Object.assign(new Event(kind), { clientX, pointerId: 1 }))

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
    expect(read.handle.getAttribute('aria-valuetext')).toBe('10 m')
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

  it('leaves no listener, observer, or frame behind once it unmounts mid-sweep', () => {
    const chart = charted(RulesSpacing, 672)
    parts(chart).player.dispatchEvent(new Event('click'))
    chart.stand.frames.run(3)
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})

describe('the track', () => {
  it.for([
    { key: 'ArrowRight', to: '15 m' },
    { key: 'ArrowUp', to: '15 m' },
    { key: 'End', to: '25 m' },
    { key: 'ArrowLeft', to: '10 m' },
    { key: 'ArrowDown', to: '10 m' },
    { key: 'Home', to: '10 m' }
  ])('steps the handle to $to on $key', ({ key, to }) => {
    const chart = charted(RulesSpacing, 672)
    expect(press(parts(chart).handle, key)).toBe(true)
    expect(parts(chart).readout).toBe(to)
  })

  it('leaves every other key to the browser', () => {
    const chart = charted(RulesSpacing, 672)
    expect(press(parts(chart).handle, 'Tab')).toBe(false)
  })

})

describe('a drag along the track', () => {
  it('reads a dragged handle between the measured spacings and settles it on the nearest once let go', () => {
    const chart   = charted(RulesSpacing, 672)
    const { hit } = parts(chart)
    captured(hit)
    pointer(hit, 'pointerdown', 377)
    expect(parts(chart).readout).toBe('17.5 m')
    expect(parts(chart).handle.getAttribute('aria-valuetext')).toBe('17.5 m, between measured spacings')
    expect(parts(chart).values[0]).toMatch(/^≈ 0\.94$/u)
    pointer(hit, 'pointermove', 9000)
    expect(parts(chart).readout).toBe('25 m')
    pointer(hit, 'pointermove', 377)
    pointer(hit, 'pointerup')
    expect(parts(chart).readout).toBe('15 m')
  })

  it('moves nothing on a pointer the strip does not hold', () => {
    const chart   = charted(RulesSpacing, 672)
    const { hit } = parts(chart)
    Object.assign(hit, { hasPointerCapture: () => false })
    pointer(hit, 'pointermove', 377)
    expect(parts(chart).readout).toBe('10 m')
  })

  it('stops a running sweep under a press on the handle', () => {
    const chart                   = charted(RulesSpacing, 672)
    const { handle, hit, player } = parts(chart)
    captured(hit)
    player.dispatchEvent(new Event('click'))
    pointer(handle, 'pointerdown')
    expect(player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })
})

describe('the sweep', () => {
  it('sweeps the handle on from the densest launch and settles it on the nearest spacing once paused', () => {
    const chart  = charted(RulesSpacing, 672)
    const player = parts(chart).player
    player.dispatchEvent(new Event('click'))
    expect(player.getAttribute('aria-pressed')).toBe('true')
    chart.stand.frames.run(12, 100)
    expect(parts(chart).readout).not.toBe('10 m')
    expect(press(player, 'Enter')).toBe(true)
    expect(player.getAttribute('aria-pressed')).toBe('false')
    expect(['10 m', '15 m', '25 m']).toContain(parts(chart).readout)
    expect(chart.stand.held().frames).toBe(0)
  })

  it('starts on a press of the space bar, leaving every other key to the browser', () => {
    const chart  = charted(RulesSpacing, 672)
    const player = parts(chart).player
    expect(press(player, 'Tab')).toBe(false)
    expect(press(player, ' ')).toBe(true)
    expect(player.getAttribute('aria-pressed')).toBe('true')
  })

  it('draws the pause glyph on a redraw while it runs', () => {
    const chart = charted(RulesSpacing, 672)
    parts(chart).player.dispatchEvent(new Event('click'))
    chart.widen(500)
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('true')
  })

  it('starts nothing for a reader who prefers reduced motion', () => {
    const chart = charted(RulesSpacing, 672, true)
    parts(chart).player.dispatchEvent(new Event('click'))
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })

})

describe('the sweep under the page’s pause', () => {
  it('settles once told the page’s pause came on', () => {
    const chart = charted(RulesSpacing, 672)
    parts(chart).player.dispatchEvent(new Event('click'))
    chart.stand.frames.run(4, 100)
    document.documentElement.dataset.paused = 'true'
    chart.stand.mutate()
    expect(parts(chart).player.getAttribute('aria-pressed')).toBe('false')
    expect(chart.stand.held().frames).toBe(0)
  })

  it('settles on its next frame once the page’s pause is on', () => {
    const chart = charted(RulesSpacing, 672)
    parts(chart).player.dispatchEvent(new Event('click'))
    document.documentElement.dataset.paused = 'true'
    chart.stand.frames.run(1)
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
