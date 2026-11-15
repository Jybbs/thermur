import { type VueWrapper, mount } from '@vue/test-utils'
import type { Component }         from 'vue'

import { type Browser, browser } from './browser.ts'

/**
 * A chart of the rules page mounted over the stand-in browser, beside the
 * stand-ins, the box it draws into, and what sets the box's width.
 */
export interface Charted {
  host   : HTMLElement
  stand  : Browser
  widen  : (width: number) => void
  widget : VueWrapper
}

/**
 * Mounts `component` into the document at `width` pixels across, its motion
 * running or standing still for a reader who prefers reduced motion, and
 * returns it beside what moves the box to another width and draws it again
 * on the next frame.
 */
export function charted(component: Component, width: number, reduced = false): Charted {
  const stand  = browser(reduced)
  const widget = mount(component, { attachTo: document.body })
  const host   = widget.element as HTMLElement
  const widen  = (next: number): void => {
    stand.reflow(host, next)
    stand.frames.run(1)
  }
  widen(width)
  return { host, stand, widen, widget }
}

/**
 * Presses `key` on `target`, returning whether the chart took it over from the
 * browser.
 */
export function press(target: Element, key: string): boolean {
  const pressed = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key })
  target.dispatchEvent(pressed)
  return pressed.defaultPrevented
}
