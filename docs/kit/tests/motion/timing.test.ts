// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { browser } from '../../../.vitepress/tests/common/browser.ts'
import * as timing from '../../src/motion/timing.ts'

/**
 * Sets the theme's settling time to `value`, as its stylesheet writes it.
 */
const settling = (value: string): void => document.documentElement.style.setProperty('--th-settle', value)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

describe('a transition', () => {
  it.for(['', '0s', 'none'])('runs once at its end where the settling time reads %j', value => {
    browser()
    settling(value)
    const step = vi.fn<(share: number) => void>()
    timing.tween(step)()
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
    timing.tween(step, scale)
    stand.frames.run(1, 120)
    expect(step.mock.lastCall![0]).toBeCloseTo(0.875, 10)
    stand.frames.run(1, 120)
    expect(step).toHaveBeenLastCalledWith(1)
    expect(stand.held().frames).toBe(0)
  })

  it('starts from no share at all on a first frame stamped before it began', () => {
    const stand = browser()
    const step  = vi.fn<(share: number) => void>()
    settling('240ms')
    vi.mocked(performance.now).mockReturnValueOnce(100)
    timing.tween(step)
    stand.frames.run(1, 16)
    expect(step).toHaveBeenLastCalledWith(0)
    expect(stand.held().frames).toBe(1)
  })

  it('stops before its end once told to', () => {
    const stand = browser()
    settling('240ms')
    const stop = timing.tween(vi.fn<(share: number) => void>())
    expect(stand.held().frames).toBe(1)
    stop()
    expect(stand.held().frames).toBe(0)
  })
})
