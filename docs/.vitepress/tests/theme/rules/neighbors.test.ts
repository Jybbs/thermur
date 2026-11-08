// @vitest-environment happy-dom
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Marks }                 from '../../../lib/canvas/marks.ts'
import { NEIGHBORS }             from '../../../lib/common.ts'
import RulesNeighbors            from '../../../theme/rules/neighbors.ts'
import { RELEASED, point }       from '../../common/browser.ts'
import { type Charted, charted } from '../../common/rules.ts'

/**
 * Where one bird stood in a frame: its middle, its length, and its color.
 */
interface Drawn {
  color  : string
  length : number
  x      : number
  y      : number
}

/**
 * Mounts the drawing over a canvas 1200 by 800 pixels on screen, its motion
 * running or standing still, and returns it beside a record of every bird it
 * draws, each by its middle.
 */
function drawn(reduced = true): Charted & { birds: Drawn[] } {
  const birds: Drawn[] = []
  vi.spyOn(Marks.prototype, 'bird').mockImplementation((_ctx, _dpr, _glyph, x, y, angle, length, color) => {
    birds.push({ color, length, x: x - Math.cos(angle) * length / 2, y: y - Math.sin(angle) * length / 2 })
  })
  const chart = charted(RulesNeighbors, 672, reduced)
  chart.stand.see(true)
  return { ...chart, birds }
}

/**
 * The birds the drawing flies over a canvas 1200 by 800 pixels, the most it
 * flies, since its sky holds more than the most at one bird per 700 square
 * pixels.
 */
const COUNT = 160

/**
 * Reads the birds the last frame over the full canvas drew, the marked bird
 * last.
 */
const frame = (birds: Drawn[]): Drawn[] => birds.slice(-COUNT)

/**
 * Finds the canvas the drawing paints.
 */
const canvas = ({ host }: Charted): HTMLCanvasElement => host.querySelector('canvas')!

/**
 * Picks the mode `value` on the switch.
 */
function picked({ host }: Charted, value: string): void {
  host.querySelector<HTMLElement>(`[data-value="${value}"]`)!.click()
}

enableAutoUnmount(afterEach)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

describe('the header', () => {
  it('names the strips, the switch, the slider, and the chip above a canvas named for what it shows', () => {
    const chart    = drawn()
    const { host } = chart
    expect(host.querySelector('.rules-title')!.textContent).toBe('Birds around the marked bird, last 12 s')
    expect([...host.querySelectorAll('[role="radio"]')].map(radio =>
      [radio.textContent, radio.getAttribute('aria-checked'), (radio as HTMLElement).title])).toEqual([
      ['Predator', 'true', 'The pointer is the falcon'],
      ['Select', 'false', 'Press a bird to mark it']
    ])
    expect(host.querySelector('input')!.getAttribute('aria-label')).toBe('Range, in bird lengths')
    expect(host.querySelector('.rules-chip-name')!.textContent).toBe('Seventh by rank')
    expect(host.querySelector('.rules-chip-value')!.textContent).toMatch(/^\d+\.\d lengths$/u)
    expect(canvas(chart).getAttribute('role')).toBe('img')
    expect(canvas(chart).getAttribute('aria-label')).toContain('over the last 12 seconds')
  })

  it('slides the switch’s thumb to the mode picked, the sky taking the mode too', () => {
    const chart = drawn()
    picked(chart, 'select')
    expect(chart.host.querySelector<HTMLElement>('.rules-mode')!.dataset.mode).toBe('select')
    expect(chart.host.querySelector<HTMLElement>('.rules-stage')!.dataset.mode).toBe('select')
  })

  it('widens the ring at the fixed range as the slider moves', async () => {
    const chart  = drawn()
    const radii  = (): number[] => chart.stand.context.arc!.mock.calls.map(([, , radius]) => radius as number)
    const before = Math.max(...radii())
    await chart.widget.get('input').setValue(6)
    expect(Math.max(...radii())).toBeCloseTo(before * 3, 6)
  })
})

describe('the marked bird', () => {
  it('draws the marked bird last, largest, with a line to each of its seven', () => {
    const chart  = drawn()
    const last   = frame(chart.birds)
    const marked = last.at(-1)!
    expect(marked.length).toBe(Math.max(...last.map(bird => bird.length)))
    expect(chart.stand.context.lineTo!.mock.calls.length).toBeGreaterThanOrEqual(NEIGHBORS)
  })

  it('moves to a bird pressed in select mode, ringing the bird under the pointer first', () => {
    const chart  = drawn()
    picked(chart, 'select')
    const before = frame(chart.birds)
    const target = before[0]!
    point(canvas(chart), { offsetX: target.x, offsetY: target.y })
    expect(chart.stand.context.arc).toHaveBeenLastCalledWith(
      expect.closeTo(target.x, 6), expect.closeTo(target.y, 6), 9, 0, Math.PI * 2)
    canvas(chart).dispatchEvent(new Event('click'))
    const marked = frame(chart.birds).at(-1)!
    expect([marked.x, marked.y]).toEqual([expect.closeTo(target.x, 6), expect.closeTo(target.y, 6)])
  })

  it('keeps its mark under a press in predator mode or away from any bird', () => {
    const chart  = drawn()
    const marked = (): Drawn => frame(chart.birds).at(-1)!
    const first  = marked()
    const other  = frame(chart.birds)[0]!
    point(canvas(chart), { offsetX: other.x, offsetY: other.y })
    canvas(chart).dispatchEvent(new Event('click'))
    expect(marked()).toEqual(first)
    picked(chart, 'select')
    point(canvas(chart), { offsetX: -500, offsetY: -500 })
    canvas(chart).dispatchEvent(new Event('click'))
    expect(marked()).toEqual(first)
  })

  it('grows the lines to a newly marked bird’s seven over the theme’s settling time', () => {
    document.documentElement.style.setProperty('--th-settle', '240ms')
    const chart  = drawn()
    picked(chart, 'select')
    const target = frame(chart.birds)[0]!
    point(canvas(chart), { offsetX: target.x, offsetY: target.y })
    canvas(chart).dispatchEvent(new Event('click'))
    expect(chart.stand.held().frames).toBe(1)
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})

describe('the strips', () => {
  it('label each strip at its end on a wide canvas and above it on a narrow one', () => {
    const chart  = drawn()
    const labels = (): unknown[][] => chart.stand.context.fillText!.mock.calls
      .filter(([words]) => String(words).startsWith('seven by rank'))
    chart.stand.context.measureText!.mockReturnValue({ width: 200 })
    chart.stand.reflow(canvas(chart))
    expect(labels().at(-1)).toEqual([`seven by rank, ${NEIGHBORS}`, 982 + 8, expect.any(Number)])
    chart.stand.resize(340, 300)
    chart.stand.reflow(canvas(chart))
    expect(labels().at(-1)).toEqual([`seven by rank, ${NEIGHBORS}`, 12, 300 - 78 + 15])
  })

  it('counts every other bird inside the ring around the marked one, the marked one left out', () => {
    const chart  = drawn()
    const last   = frame(chart.birds)
    const marked = last.at(-1)!
    const reach  = Math.max(...chart.stand.context.arc!.mock.calls.map(([, , radius]) => radius as number))
    const within = last.slice(0, -1)
      .filter(bird => Math.hypot(bird.x - marked.x, bird.y - marked.y) <= reach).length
    const counts = chart.stand.context.fillText!.mock.calls
      .map(([words]) => String(words))
      .filter(words => words.startsWith('within range'))
    expect(within).toBeGreaterThan(0)
    expect(counts.at(-1)).toBe(`within range, ${within}`)
  })

  it('counts the birds inside the range as the flock flies under the falcon, leaving nothing behind', () => {
    const chart  = drawn(false)
    const before = frame(chart.birds)
    point(canvas(chart), { offsetX: 600, offsetY: 300 })
    chart.stand.frames.run(40, 50)
    canvas(chart).dispatchEvent(new Event('pointerleave'))
    chart.stand.frames.run(5, 50)
    expect(frame(chart.birds)).not.toEqual(before)
    expect(chart.stand.context.fillText).toHaveBeenCalledWith(
      expect.stringMatching(/^within range, \d+$/u), expect.any(Number), expect.any(Number))
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})
