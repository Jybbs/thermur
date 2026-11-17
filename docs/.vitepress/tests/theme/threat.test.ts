// @vitest-environment happy-dom
import { Slider }                          from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import ThreatFront                         from '../../theme/threat.ts'
import { type Browser, RELEASED, browser } from '../common/browser.ts'

/**
 * A mounted drawing over the stand-in browser, beside the stand-ins.
 */
interface Mounted {
  stand  : Browser
  widget : VueWrapper
}

/**
 * Mounts the drawing into the document with the frames running, or standing
 * still for a reader who prefers reduced motion.
 */
function mounted(reduced = false): Mounted {
  const stand  = browser(reduced)
  const widget = mount(ThreatFront, { attachTo: document.body })
  stand.see(true)
  return { stand, widget }
}

/**
 * Reads the chips beneath the drawing, each its name and its value.
 */
const chips = (widget: VueWrapper): string[] => widget.findAll('.kit-chip').map(chip => chip.text())

/**
 * Moves the rail to `seconds` as a drag along it does.
 */
async function drag(widget: VueWrapper, seconds: number): Promise<void> {
  widget.getComponent(Slider).vm.$emit('change', seconds)
  await widget.vm.$nextTick()
}

/**
 * Presses `key` on the rail's thumb, once the thumb holds the focus, and
 * returns the event pressed.
 */
async function press(widget: VueWrapper, key: string): Promise<KeyboardEvent> {
  await widget.vm.$nextTick()
  const thumb   = widget.get('[role=slider]').element as HTMLElement
  const pressed = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key })
  thumb.focus()
  thumb.dispatchEvent(pressed)
  await widget.vm.$nextTick()
  return pressed
}

/**
 * The rail as a case reaches it, through its component.
 */
type Rail = Pick<VueWrapper, 'trigger' | 'vm'>

/**
 * The ways a press on the rail ends: the release that commits where it stands,
 * and a cancel the browser sends.
 */
const RELEASES = [
  {
    end     : 'commit',
    release : (rail: Rail): Promise<void> => {
      rail.vm.$emit('commit', 30)
      return rail.vm.$nextTick()
    }
  },
  { end: 'pointercancel', release: (rail: Rail): Promise<void> => rail.trigger('pointercancel') }
]

enableAutoUnmount(afterEach)

afterEach(() => {
  delete document.documentElement.dataset.paused
})

describe('the drawing of the front', () => {
  it('sets its canvas in an image named for what it shows, its rail named for the time', () => {
    const { widget } = mounted(true)
    expect(widget.get('[role=img]').attributes('aria-label')).toContain('a front of hot air')
    expect(widget.find('canvas').exists()).toBe(true)
    expect(widget.get('[role=slider]').attributes()).toMatchObject({
      'aria-label'    : 'Time since the first snapshot',
      'aria-valuemax' : '120',
      'aria-valuemin' : '0'
    })
  })

  it('names the time, the front, and the first and last cell beneath it', () => {
    const { widget } = mounted(true)
    expect(chips(widget)).toEqual(['Time60 s', 'Front60 m', 'First cell317.5 K', 'Last cell299.9 K'])
  })

  it('leaves no listener, observer, animation frame, or canvas behind once it unmounts', () => {
    const { stand, widget } = mounted()
    stand.frames.run(3)
    expect(stand.held().canvases).toBe(1)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('stands still while the page’s pause is on, and runs again once it is off', () => {
    const { stand, widget } = mounted()
    stand.frames.run(50, 50)
    const stopped = chips(widget)
    document.documentElement.dataset.paused = 'true'
    stand.mutate()
    stand.frames.run(50, 50)
    expect(stand.held().frames).toBe(0)
    expect(chips(widget)).toEqual(stopped)
    document.documentElement.dataset.paused = 'false'
    stand.mutate()
    stand.frames.run(50, 50)
    expect(chips(widget)).not.toEqual(stopped)
  })
})

describe('the rail', () => {
  it('moves a drawing standing still at once, the reader’s time read back in the chips', async () => {
    const { widget } = mounted(true)
    await drag(widget, 90)
    expect(chips(widget)[0]).toBe('Time90 s')
    expect(widget.get('[role=slider]').attributes('aria-valuetext')).toBe('90 seconds, the front 90 meters east')
  })

  it.for([
    { key: 'ArrowRight', time: 'Time75 s' },
    { key: 'ArrowUp', time: 'Time75 s' },
    { key: 'ArrowLeft', time: 'Time45 s' },
    { key: 'ArrowDown', time: 'Time45 s' },
    { key: 'Home', time: 'Time0 s' },
    { key: 'End', time: 'Time120 s' }
  ])('takes $key to $time, moving nothing else on the page', async ({ key, time }) => {
    const { widget } = mounted(true)
    const pressed    = await press(widget, key)
    expect(chips(widget)[0]).toBe(time)
    expect(pressed.defaultPrevented).toBe(true)
  })

  it('leaves every other key to the browser', async () => {
    const { widget } = mounted(true)
    const pressed    = await press(widget, 'Tab')
    expect(pressed.defaultPrevented).toBe(false)
    expect(chips(widget)[0]).toBe('Time60 s')
  })

  it.for(RELEASES)('holds the run under a press and lets go on its $end', async ({ release }) => {
    const { stand, widget } = mounted()
    const rail              = widget.getComponent(Slider)
    await rail.trigger('pointerdown')
    await drag(widget, 30)
    stand.frames.run(400, 50)
    expect(chips(widget)[0]).toBe('Time30 s')
    await release(rail)
    stand.frames.run(70, 50)
    expect(chips(widget)[0]).toBe('Time30 s')
    stand.frames.run(20, 50)
    expect(chips(widget)[0]).not.toBe('Time30 s')
  })
})

describe('the rail over a drawing already drawn', () => {
  it('repaints a drawing standing still on a move once its face has loaded', async () => {
    const { widget } = mounted(true)
    await flushPromises()
    await drag(widget, 90)
    expect(chips(widget)[0]).toBe('Time90 s')
  })

  it('holds the run after a press that moved nothing', async () => {
    const { stand, widget } = mounted()
    stand.frames.run(10, 50)
    const rail = widget.getComponent(Slider)
    await rail.trigger('pointerdown')
    rail.vm.$emit('commit', Math.round(Number(rail.props('value'))))
    const held = chips(widget)[0]
    stand.frames.run(70, 50)
    expect(chips(widget)[0]).toBe(held)
  })
})
