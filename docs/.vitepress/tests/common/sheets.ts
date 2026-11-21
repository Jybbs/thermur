import { Window } from 'happy-dom'

import label from '../../../kit/src/styles/label.css?raw'

/**
 * The face, weight, spacing, and case every control's label is set at.
 */
export const CONTROL_LABEL = {
  'font-family'    : 'var(--vp-font-family-mono)',
  'font-weight'    : '500',
  'letter-spacing' : '0.1em',
  'text-transform' : 'uppercase'
} as const

/**
 * Opens a page holding the stylesheet whose text is `sheet`, for a reader who
 * prefers reduced motion where `reduced` holds.
 */
function opened(sheet: string, reduced: boolean = false): { document: Document, style: HTMLStyleElement } {
  const motion       = reduced ? 'reduce' : 'no-preference'
  const { document } = new Window({ settings: { device: { prefersReducedMotion: motion } } })
  const style        = document.createElement('style')
  style.textContent = sheet
  document.head.append(style)
  return { document: document as unknown as Document, style: style as unknown as HTMLStyleElement }
}

/**
 * Reads the computed value of `property` on the element `selector` finds in
 * `html`, once the stylesheet whose text is `sheet` is loaded, for a reader who
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
 * Reads the value the stylesheet whose text is `sheet` declares for `property`
 * in the rule whose selector list is `selector`, for a state such as `:active`
 * that no computed style reaches.
 */
export function declared(property: string, selector: string, sheet: string): string {
  const rules = [...opened(sheet).style.sheet!.cssRules] as CSSStyleRule[]
  return rules.find(rule => rule.selectorText?.replaceAll(/\s+/gu, ' ') === selector)!.style.getPropertyValue(property)
}

/**
 * Lists every style rule in `list`, those inside an at-rule such as `@media`
 * included.
 */
function flattened(list: CSSRuleList): CSSStyleRule[] {
  return [...list].flatMap(rule =>
    'selectorText' in rule ? [rule as CSSStyleRule] : 'cssRules' in rule ? flattened((rule as CSSGroupingRule).cssRules) : []
  )
}

/**
 * Reads the face, weight, spacing, and case `element` is set in, each from its
 * inline style where it sets one and otherwise from the kit's label rule, or
 * nothing where none of the rule's selectors matches it.
 */
export function labeled(element: SVGElement): Record<string, string> {
  const [rule] = flattened(opened(label).style.sheet!.cssRules)
  const keys   = Object.keys(CONTROL_LABEL)
  return element.matches(rule!.selectorText)
    ? Object.fromEntries(keys.map(key => [
      key, element.style.getPropertyValue(key) || rule!.style.getPropertyValue(key)
    ]))
    : {}
}

/**
 * Lists every style rule the stylesheet whose text is `sheet` holds, those
 * inside an at-rule such as `@media` included.
 */
export function styled(sheet: string): CSSStyleRule[] {
  return flattened(opened(sheet).style.sheet!.cssRules)
}
