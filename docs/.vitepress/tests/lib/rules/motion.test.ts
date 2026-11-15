// @vitest-environment happy-dom
import { type Mock, afterEach, describe, expect, it, vi } from 'vitest'

import * as motion                             from '../../../lib/rules/motion.ts'
import { type Browser, RELEASED, box, browser } from '../../common/browser.ts'

/**
 * Sets the theme's settling time to `value`, as its stylesheet writes it.
 */
const settling = (value: string): void => document.documentElement.style.setProperty('--th-settle', value)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
  delete document.documentElement.dataset.paused
})

describe('a transition', () => {
  it.for(['', '0s', 'none'])('runs once at its end where the settling time reads %j', value => {
    browser()
    settling(value)
    const step = vi.fn<(share: number) => void>()
    motion.tween(step)()
    expect(step).toHaveBeenCalledExactlyOnceWith(1)
  })

  it.for([
    { scale: 1, value: '240ms' },
    { scale: 2, value: '0.12s' }
  ])('eases over $value times $scale, slowing into its end', ({ scale, value }) => {
    const stand = browser()
    const step  = vi.fn<(share: number) => void>()
    vi.spyOn(performance, 'now').mockReturnValue(0)
    settling(value)
    motion.tween(step, scale)
    stand.frames.run(1, 120)
    expect(step.mock.lastCall![0]).toBeCloseTo(0.875, 10)
    stand.frames.run(1, 120)
    expect(step).toHaveBeenLastCalledWith(1)
    expect(stand.held().frames).toBe(0)
  })

  it('stops before its end once told to', () => {
    const stand = browser()
    settling('240ms')
    const stop = motion.tween(vi.fn<(share: number) => void>())
    expect(stand.held().frames).toBe(1)
    stop()
    expect(stand.held().frames).toBe(0)
  })
})

describe('the motion a chart reads', () => {
  it('stands still for a reader who prefers reduced motion or while the page’s pause is on', () => {
    const stand = browser()
    expect(motion.still()).toBe(false)
    document.documentElement.dataset.paused = 'true'
    expect(motion.still()).toBe(true)
    delete document.documentElement.dataset.paused
    stand.motion.matches = true
    expect(motion.still()).toBe(true)
  })

  it('calls back on a change of the pause or the preference, until released', () => {
    const stand   = browser()
    const changed = vi.fn<() => void>()
    const release = motion.onMotion(changed)
    stand.mutate()
    for (const listener of stand.motion.listeners) listener()
    expect(changed).toHaveBeenCalledTimes(2)
    release()
    expect(stand.held()).toEqual(RELEASED)
  })
})

/**
 * Watches the width of a fresh box over the stand-ins, returning the stand-ins,
 * the box, the callback the width reaches, and what releases the watch.
 */
function watched(): { changed: Mock<(width: number) => void>, host: HTMLElement, release: () => void, stand: Browser } {
  const stand   = browser()
  const host    = box('rules-host')
  const changed = vi.fn<(width: number) => void>()
  return { changed, host, release: motion.widthOf(host, changed), stand }
}

describe('the width a chart is drawn at', () => {
  it('calls back with a width only once the box takes one it was not drawn at', () => {
    const { changed, host, release, stand } = watched()
    for (const width of [0, 640, 640.3, 390]) {
      stand.reflow(host, width)
      stand.frames.run(1)
    }
    expect(changed.mock.calls).toEqual([[640], [390]])
    release()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('draws at a new width on the next frame, measuring nothing inside the observer’s callback', () => {
    const measured                 = vi.spyOn(Element.prototype, 'getBoundingClientRect')
    const { changed, host, stand } = watched()
    stand.reflow(host, 640)
    expect(changed).not.toHaveBeenCalled()
    stand.frames.run(1)
    expect(changed).toHaveBeenCalledExactlyOnceWith(640)
    expect(measured).not.toHaveBeenCalled()
  })

  it('draws once at the last width a frame sees, dropping the frame on release', () => {
    const { changed, host, release, stand } = watched()
    stand.reflow(host, 640)
    stand.reflow(host, 390)
    stand.frames.run(1)
    expect(changed.mock.calls).toEqual([[390]])
    stand.reflow(host, 512)
    release()
    expect(stand.held()).toEqual(RELEASED)
    stand.frames.run(1)
    expect(changed).toHaveBeenCalledOnce()
  })

  it('draws at the width a box returns to within one frame after leaving it', () => {
    const { changed, host, stand } = watched()
    stand.reflow(host, 640)
    stand.frames.run(1)
    stand.reflow(host, 390)
    stand.reflow(host, 640)
    stand.frames.run(1)
    expect(changed.mock.calls.at(-1)).toEqual([640])
  })
})
