// @vitest-environment happy-dom
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it }           from 'vitest'

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
const chips = (widget: VueWrapper): string[] => widget.findAll('.threat-chip').map(chip => chip.text())

enableAutoUnmount(afterEach)

afterEach(() => {
  delete document.documentElement.dataset.paused
})

describe('the drawing of the front', () => {
  it('sets its canvas in an image named for what it shows, its rail named for the time', () => {
    const { widget } = mounted(true)
    expect(widget.get('[role=img]').attributes('aria-label')).toContain('a front of hot air')
    expect(widget.find('canvas').exists()).toBe(true)
    expect(widget.get('input').attributes()).toMatchObject({
      'aria-label' : 'Time since the first snapshot',
      'max'        : '120',
      'min'        : '0',
      'type'       : 'range'
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
    await widget.get('input').setValue(90)
    expect(chips(widget)[0]).toBe('Time90 s')
    expect(widget.get('input').attributes('aria-valuetext')).toBe('90 seconds, the front 90 meters east')
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
    const pressed    = new KeyboardEvent('keydown', { cancelable: true, key })
    widget.get('input').element.dispatchEvent(pressed)
    await widget.vm.$nextTick()
    expect(chips(widget)[0]).toBe(time)
    expect(pressed.defaultPrevented).toBe(true)
  })

  it('leaves every other key to the browser', () => {
    const { widget } = mounted(true)
    const pressed    = new KeyboardEvent('keydown', { cancelable: true, key: 'Tab' })
    widget.get('input').element.dispatchEvent(pressed)
    expect(pressed.defaultPrevented).toBe(false)
    expect(chips(widget)[0]).toBe('Time60 s')
  })

  it.for(['pointerup', 'pointercancel'])('holds the run under a press and lets go on %s', async end => {
    const { stand, widget } = mounted()
    const rail              = widget.get('input')
    await rail.trigger('pointerdown')
    await rail.setValue(30)
    stand.frames.run(400, 50)
    expect(chips(widget)[0]).toBe('Time30 s')
    await rail.trigger(end)
    stand.frames.run(70, 50)
    expect(chips(widget)[0]).toBe('Time30 s')
    stand.frames.run(20, 50)
    expect(chips(widget)[0]).not.toBe('Time30 s')
  })
})
