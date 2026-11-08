// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest'

import * as motion                from '../../../lib/rules/motion.ts'
import { RELEASED, box, browser } from '../../common/browser.ts'

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

  it('calls back with a width only once the box takes one it was not drawn at', () => {
    const stand   = browser()
    const host    = box('rules-host')
    const widths  = [0, 640, 640.3, 390]
    const changed = vi.fn<(width: number) => void>()
    vi.spyOn(host, 'getBoundingClientRect').mockImplementation(() => ({ width: widths.shift() }) as DOMRect)
    const release = motion.widthOf(host, changed)
    for (let k = 0; k < 4; k++) stand.reflow(host)
    expect(changed.mock.calls).toEqual([[640], [390]])
    release()
    expect(stand.held()).toEqual(RELEASED)
  })
})
