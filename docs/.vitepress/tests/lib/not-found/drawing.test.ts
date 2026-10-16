// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { drawLost }                          from '../../../lib/not-found/drawing.ts'
import { Opening }                           from '../../../lib/not-found/opening.ts'
import { rimOf }                             from '../../../lib/not-found/passing.ts'
import { type Browser, box, browser, point } from '../../common/browser.ts'

/**
 * The oval around the numerals at rest on the stand-in's canvas, 1200 by 800
 * pixels.
 */
const AT_REST = { a: 136, b: 96, cx: 600, cy: 400 }

/**
 * Runs one frame and reads how far outside the oval around the numerals at rest
 * the closest bird it placed flies.
 */
function closest(stand: Browser): number {
  vi.clearAllMocks()
  stand.frames.run(1)
  return Math.min(...placed(stand).map(call => rim(call)))
}

/**
 * Reads the transform of each bird the stand-in context placed, leaving out the
 * plain scaling a paint resets the context to.
 */
const placed = (stand: Browser): unknown[][] =>
  stand.context.setTransform!.mock.calls.filter(([a, b]) => a !== 1 || b !== 0)

/**
 * Moves the pointer to `x` and `y` over the drawing's canvas.
 */
const rest = (x: number, y: number): void =>
  point(document.querySelector('canvas')!, { offsetX: x, offsetY: y })

/**
 * Measures how far outside the oval around the numerals at rest a bird the
 * stand-in context placed flies, 1 on its rim.
 */
const rim = ([, , , , x, y]: unknown[]): number => rimOf(AT_REST, Number(x), Number(y))

/**
 * Counts the hairlines the stand-in context strokes over `count` frames.
 */
function stroked(stand: Browser, count: number): number {
  const before = stand.context.stroke?.mock.calls.length ?? 0
  stand.frames.run(count)
  return (stand.context.stroke?.mock.calls.length ?? 0) - before
}

describe('the drawing of the page met at a missing address', () => {
  it('builds no element, gradient, or path on a run of frames after the first fit', () => {
    const stand = browser()
    const stage = drawLost(box('not-found-stage'))
    stand.see(true)
    rest(200, 400)
    stand.frames.run(2)
    const built = stand.built()
    stand.frames.run(30)
    expect(stand.built()).toEqual(built)
    stage.destroy()
  })

  it('draws its birds and numerals in the dark scheme’s tones apart from the light scheme’s', () => {
    const stand = browser(true)
    const stage = drawLost(box('not-found-stage'))
    const light = stand.wrote('fillStyle')
    document.documentElement.classList.add('dark')
    stage.refit()
    document.documentElement.classList.remove('dark')
    const dark = stand.wrote('fillStyle').slice(light.length)
    expect(dark.at(-1)).not.toBe(light.at(-1))
    expect(new Set(dark.slice(0, -1))).not.toEqual(new Set(light.slice(0, -1)))
    stage.destroy()
  })

  it('fits again once the display face has loaded', async () => {
    const stand = browser()
    const fit   = vi.spyOn(Opening.prototype, 'fit')
    const stage = drawLost(box('not-found-stage'))
    expect(stand.fonts.load).toHaveBeenCalledWith("400 1em 'Bellefair'")
    await vi.waitFor(() => expect(fit).toHaveBeenCalledTimes(2))
    stage.destroy()
  })
})

describe('the flock on the page met at a missing address', () => {
  it('carries each bird past the right edge back in at the left', () => {
    const stand = browser()
    const stage = drawLost(box('not-found-stage'))
    stand.see(true)
    stand.frames.run(24, 50)
    vi.clearAllMocks()
    stand.frames.run(1, 50)
    const across = placed(stand).map(([, , , , x]) => Number(x))
    expect(Math.max(...across)).toBeLessThanOrEqual(1220)
    expect(Math.min(...across)).toBeGreaterThanOrEqual(-20)
    expect(Math.min(...across)).toBeLessThan(0)
    stage.destroy()
  })

  it('lays 45% of the flock across a narrow page, each bird shorter than on a wide one', () => {
    const stand = browser(true)
    stand.resize(390, 400)
    const stage = drawLost(box('not-found-stage'))
    const birds = placed(stand)
    expect(birds).toHaveLength(Math.round(0.45 * 1500))
    expect(Math.min(...birds.map(([a, b]) => Math.hypot(Number(a), Number(b))))).toBeLessThan(3.9)
    stage.destroy()
  })
})

describe('the seven links and the burst on the page met at a missing address', () => {
  it('shows the seven links only once the pointer has rested, and draws them back as it moves', () => {
    const stand = browser()
    const stage = drawLost(box('not-found-stage'))
    stand.see(true)
    rest(200, 400)
    expect(stroked(stand, 20)).toBe(0)
    stand.frames.run(60)
    expect(stroked(stand, 1)).toBe(7)
    rest(320, 360)
    stand.frames.run(30)
    expect(stroked(stand, 1)).toBe(0)
    stage.destroy()
  })

  it('shows the seven at once on a resting pointer while the drawing stands still', () => {
    const stand = browser(true)
    const stage = drawLost(box('not-found-stage'))
    expect(stand.context.stroke).toBeUndefined()
    rest(200, 400)
    expect(stand.context.stroke!.mock.calls).toHaveLength(7)
    stage.destroy()
  })

  it('bursts the hole around the numerals wider for a moment on a click', () => {
    const stand = browser()
    const stage = drawLost(box('not-found-stage'))
    stand.see(true)
    stand.frames.run(27)
    const resting = closest(stand)
    document.querySelector('canvas')!.dispatchEvent(new Event('click'))
    stand.frames.run(27)
    expect(closest(stand)).toBeGreaterThan(resting * 1.3)
    stand.frames.run(39)
    expect(closest(stand)).toBeCloseTo(resting, 1)
    stage.destroy()
  })
})
