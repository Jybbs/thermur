import fs from 'node:fs'

import { Window } from 'happy-dom'

/**
 * Opens a page holding the theme's stylesheet `sheet`, for a reader who prefers
 * reduced motion where `reduced` holds.
 */
function opened(sheet: string, reduced: boolean = false): { document: Document, style: HTMLStyleElement } {
  const motion       = reduced ? 'reduce' : 'no-preference'
  const { document } = new Window({ settings: { device: { prefersReducedMotion: motion } } })
  const style        = document.createElement('style')
  style.textContent = fs.readFileSync(new URL(`../../theme/styles/${sheet}`, import.meta.url), 'utf8')
  document.head.append(style)
  return { document: document as unknown as Document, style: style as unknown as HTMLStyleElement }
}

/**
 * Reads the computed value of `property` on the element `selector` finds in
 * `html`, once the theme's stylesheet `sheet` is loaded, for a reader who
 * prefers reduced motion where `reduced` holds.
 */
export function computed(
  html     : string,
  property : string,
  selector : string,
  sheet    : string,
  reduced  : boolean = false
): string {
  const { document }      = opened(sheet, reduced)
  document.body.innerHTML = html
  return document.defaultView!.getComputedStyle(document.querySelector(selector)!).getPropertyValue(property)
}

/**
 * Reads the value the theme's stylesheet `sheet` declares for `property` in the
 * rule whose selector list is `selector`, for a state such as `:active` that no
 * computed style reaches.
 */
export function declared(property: string, selector: string, sheet: string): string {
  const rules = [...opened(sheet).style.sheet!.cssRules] as CSSStyleRule[]
  return rules.find(rule => rule.selectorText?.replaceAll(/\s+/gu, ' ') === selector)!.style.getPropertyValue(property)
}
