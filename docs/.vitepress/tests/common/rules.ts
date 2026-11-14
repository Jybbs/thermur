import { type VueWrapper, mount } from '@vue/test-utils'
import { vi }                     from 'vitest'
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
 * returns it beside what moves every box to another width and draws it again.
 */
export function charted(component: Component, width: number, reduced = false): Charted {
  const stand = browser(reduced)
  let across  = width
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() =>
    ({ height: 300, left: 0, top: 0, width: across }) as DOMRect)
  const widget = mount(component, { attachTo: document.body })
  const host   = widget.element as HTMLElement
  const widen  = (next: number): void => {
    across = next
    stand.reflow()
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
