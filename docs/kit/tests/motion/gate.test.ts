// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RELEASED, browser } from '../../../.vitepress/tests/common/browser.ts'
import * as gate             from '../../src/motion/gate.ts'

afterEach(() => {
  delete document.documentElement.dataset.paused
})

describe('the gate a moving drawing passes', () => {
  it('stands still for a reader who prefers reduced motion or while the page’s pause is on', () => {
    const stand = browser()
    expect(gate.still()).toBe(false)
    document.documentElement.dataset.paused = 'true'
    expect(gate.still()).toBe(true)
    delete document.documentElement.dataset.paused
    stand.motion.matches = true
    expect(gate.still()).toBe(true)
  })

  it('calls back on a change of the pause or the preference, until released', () => {
    const stand   = browser()
    const changed = vi.fn<() => void>()
    const release = gate.onMotion(changed)
    stand.mutate()
    for (const listener of stand.motion.listeners) listener()
    expect(changed).toHaveBeenCalledTimes(2)
    release()
    expect(stand.held()).toEqual(RELEASED)
  })


  it('turns the page’s pause on and off on the root element', () => {
    gate.pause(true)
    expect([document.documentElement.dataset.paused, gate.paused()]).toEqual(['true', true])
    gate.pause(false)
    expect([document.documentElement.dataset.paused, gate.paused()]).toEqual(['false', false])
  })
})
