// @vitest-environment happy-dom
import { pause }                                     from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it }           from 'vitest'
import { defineComponent }                           from 'vue'

import { type Loop, useLoop } from '../../../lib/kit/loop.ts'
import { RELEASED, browser }  from '../../common/browser.ts'

enableAutoUnmount(afterEach)

afterEach(() => {
  delete document.documentElement.dataset.paused
})

/**
 * Mounts a component running a loop that writes down the seconds of each
 * frame it is called with.
 */
function looped(): { loop: Loop, seen: number[], widget: VueWrapper } {
  const seen = [] as number[]
  let loop!: Loop
  const widget = mount(defineComponent({
    setup() {
      loop = useLoop(seconds => seen.push(seconds))
      return () => null
    }
  }))
  return { loop, seen, widget }
}

describe('useLoop', () => {
  it('calls its frame with the seconds since the last one while it runs, and stops when told', () => {
    const stand          = browser()
    const { loop, seen } = looped()
    loop.run(true)
    stand.frames.run(3, 50)
    expect(loop.running.value).toBe(true)
    expect(seen).toHaveLength(3)
    expect(seen.at(-1)).toBeCloseTo(0.05)
    loop.run(false)
    stand.frames.run(2, 50)
    expect([loop.running.value, seen.length]).toEqual([false, 3])
  })

  it('runs nothing for a reader who prefers reduced motion', () => {
    const stand          = browser(true)
    const { loop, seen } = looped()
    loop.run(true)
    stand.frames.run(2)
    expect([loop.standing.value, loop.running.value, seen.length]).toEqual([true, false, 0])
  })

  it('stops as the page’s pause turns on, and stands still until it turns off', () => {
    const stand          = browser()
    const { loop, seen } = looped()
    loop.run(true)
    pause(true)
    stand.mutate('data-paused')
    stand.frames.run(2)
    expect([loop.standing.value, loop.running.value, seen.length]).toEqual([true, false, 0])
    pause(false)
    stand.mutate('data-paused')
    expect(loop.standing.value).toBe(false)
  })

  it('releases every frame and listener once its component unmounts', () => {
    const stand            = browser()
    const { loop, widget } = looped()
    loop.run(true)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})
