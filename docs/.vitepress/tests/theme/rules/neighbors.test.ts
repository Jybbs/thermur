// @vitest-environment happy-dom
import { Slider }                              from '@thermur/kit'
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Marks }                 from '../../../lib/canvas/marks.ts'
import { FLIGHT, Flock }         from '../../../lib/front/flock.ts'
import { seriesOf }              from '../../../lib/identity.ts'
import { RANGED }                from '../../../lib/rules/falcon.ts'
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
 * Picks the mode `value` among the radios.
 */
function picked({ host }: Charted, value: string): void {
  host.querySelector<HTMLInputElement>(`input[type=radio][value="${value}"]`)!.click()
}

/**
 * Reads each radio's words, whether it is on, and its title.
 */
const radios = ({ host }: Charted): [string, boolean, string][] =>
  [...host.querySelectorAll<HTMLElement>('.kit-radios-item')].map(item =>
    [item.textContent, item.querySelector('input')!.checked, item.title])

/**
 * Cuts the senders `Flock.heard` writes to the first `count` each bird ranked,
 * `-1` past them, as for a bird whose radio holds fewer than seven.
 */
function short(count: number): void {
  const heard = Flock.prototype.heard
  vi.spyOn(Flock.prototype, 'heard').mockImplementation(function cut(this: Flock, index, into) {
    heard.call(this, index, into)
    into.fill(-1, count)
    return count
  })
}

/**
 * Mounts the drawing and reads the scale each bird of its last frame is drawn
 * at, its length over the size its flock gives it, in the order the frame
 * draws them, the marked bird last.
 */
function scaled(): number[] {
  const marked = { chosen: -1, flock: undefined as Flock | undefined }
  const heard  = Flock.prototype.heard
  vi.spyOn(Flock.prototype, 'heard').mockImplementation(function seen(this: Flock, index, into) {
    marked.chosen = index
    marked.flock  = this
    return heard.call(this, index, into)
  })
  const last              = frame(drawn().birds)
  const { chosen, flock } = marked
  const order             = [...last.keys()].filter(index => index !== chosen).concat(chosen)
  return last.map((bird, at) => bird.length / flock!.size[order[at]!]!)
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
    expect(radios(chart)).toEqual([
      ['Predator', true, 'The pointer is the falcon'],
      ['Select', false, 'Press a bird to mark it']
    ])
    expect(host.querySelector('[role=slider]')!.getAttribute('aria-label')).toBe('Range, in bird lengths')
    expect(host.querySelector('.kit-chip-name')!.textContent).toBe('Farthest ranked')
    expect(host.querySelector('.kit-chip-value')!.textContent).toMatch(/^\d+\.\d lengths$/u)
    expect(canvas(chart).getAttribute('role')).toBe('img')
    expect(canvas(chart).getAttribute('aria-label')).toContain('over the last 12 seconds')
  })

  it('turns on the mode picked, the sky taking the mode too', async () => {
    const chart = drawn()
    picked(chart, 'select')
    await chart.widget.vm.$nextTick()
    expect(radios(chart).map(([, on]) => on)).toEqual([false, true])
    expect(chart.host.querySelector<HTMLElement>('.rules-stage')!.dataset.mode).toBe('select')
  })

  it('widens the ring at the fixed range as the slider moves', () => {
    const chart  = drawn()
    const radii  = (): number[] => chart.stand.context.arc!.mock.calls.map(([, , radius]) => radius as number)
    const before = Math.max(...radii())
    chart.widget.getComponent(Slider).vm.$emit('change', 6)
    expect(Math.max(...radii())).toBeCloseTo(before * 2, 6)
  })
})

describe('the marked bird', () => {
  it('draws the marked bird last and at the largest scale of its size', () => {
    const scales = scaled()
    expect(scales.at(-1)! - Math.max(...scales.slice(0, -1))).toBeGreaterThan(1e-9)
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

describe('the lines and the chip', () => {
  it('draw to the senders the marked bird’s last step ranked, the chip reading the last one’s distance', () => {
    const heard           = vi.spyOn(Flock.prototype, 'heard')
    const chart           = drawn()
    const flock           = heard.mock.contexts.at(-1)! as Flock
    const [chosen, ranks] = heard.mock.lastCall!
    const marked          = frame(chart.birds).at(-1)!
    const senders         = [...ranks].filter(index => index >= 0)
    const zoom            = marked.x / flock.x[chosen]!
    const length          = (FLIGHT.size[0] + FLIGHT.size[1]) / 2
    expect(flock.y[chosen]! * zoom).toBeCloseTo(marked.y, 6)
    expect(chart.stand.context.lineTo!.mock.calls.slice(-senders.length)).toEqual(senders.map(index =>
      [expect.closeTo(flock.x[index]! * zoom, 6), expect.closeTo(flock.y[index]! * zoom, 6)]))
    expect(chart.host.querySelector('.kit-chip-value')!.textContent)
      .toBe(`${(flock.apart(senders.at(-1)!, chosen) / length).toFixed(1)} lengths`)
  })
})

describe('a marked bird ranking fewer than seven', () => {
  it('draws a line to each sender it ranks, the chip reading the last one’s distance', () => {
    short(3)
    const chart           = drawn()
    const heard           = vi.mocked(Flock.prototype.heard)
    const flock           = heard.mock.contexts.at(-1)! as Flock
    const [chosen, ranks] = heard.mock.lastCall!
    const length          = (FLIGHT.size[0] + FLIGHT.size[1]) / 2
    chart.stand.context.lineTo!.mockClear()
    chart.stand.reflow(canvas(chart))
    expect(chart.stand.context.lineTo).toHaveBeenCalledTimes(3)
    expect(chart.host.querySelector('.kit-chip-value')!.textContent)
      .toBe(`${(flock.apart(ranks[2]!, chosen) / length).toFixed(1)} lengths`)
  })

  it('reads 0 on the chip where it ranks no one', () => {
    short(0)
    expect(drawn().host.querySelector('.kit-chip-value')!.textContent).toBe('0.0 lengths')
  })

  it('counts by rank the senders it ranks, held and flying, until it ranks seven again', () => {
    short(3)
    const chart  = drawn(false)
    const labels = (): string[] => chart.stand.context.fillText!.mock.calls
      .map(([words]) => String(words))
      .filter(words => words.startsWith('BY RANK'))
    expect(labels().at(-1)).toBe('BY RANK, 3')
    chart.stand.frames.run(20, 50)
    expect(labels().at(-1)).toBe('BY RANK, 3')
    vi.mocked(Flock.prototype.heard).mockRestore()
    chart.stand.frames.run(20, 50)
    expect(labels().at(-1)).toBe('BY RANK, 7')
  })
})

describe('the strips', () => {
  it('label each strip at its end on a wide canvas and above it on a narrow one, by the count ranked', () => {
    const heard  = vi.spyOn(Flock.prototype, 'heard')
    const chart  = drawn()
    const ranked = (): string => `BY RANK, ${heard.mock.results.at(-1)!.value}`
    const labels = (): unknown[][] => chart.stand.context.fillText!.mock.calls
      .filter(([words]) => String(words).startsWith('BY RANK'))
    chart.stand.context.measureText!.mockReturnValue({ width: 200 })
    chart.stand.reflow(canvas(chart))
    expect(labels().at(-1)).toEqual([ranked(), 982 + 8, expect.any(Number)])
    chart.stand.resize(340, 300)
    chart.stand.reflow(canvas(chart))
    expect(labels().at(-1)).toEqual([ranked(), 12, 300 - 78 + 15])
  })

  it('counts every bird it draws in the range’s color, each inside the ring, the marked one left out', () => {
    const chart  = drawn()
    const last   = frame(chart.birds)
    const marked = last.at(-1)!
    const reach  = Math.max(...chart.stand.context.arc!.mock.calls.map(([, , radius]) => radius as number))
    const ranged = new Set((['dark', 'light'] as const).map(scheme => seriesOf(RANGED, scheme)))
    const inside = last.slice(0, -1).filter(bird => ranged.has(bird.color))
    const counts = chart.stand.context.fillText!.mock.calls
      .map(([words]) => String(words))
      .filter(words => words.startsWith('WITHIN RANGE'))
    expect(inside.length).toBeGreaterThan(0)
    for (const bird of inside) {
      expect(Math.hypot(bird.x - marked.x, bird.y - marked.y)).toBeLessThanOrEqual(reach)
    }
    expect(counts.at(-1)).toBe(`WITHIN RANGE, ${inside.length}`)
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
      expect.stringMatching(/^WITHIN RANGE, \d+$/u), expect.any(Number), expect.any(Number))
    chart.widget.unmount()
    expect(chart.stand.held()).toEqual(RELEASED)
  })
})
