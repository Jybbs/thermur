import { describe, expect, it } from 'vitest'

import { declared, styled } from '../../common/sheets.ts'

/**
 * The links and the chips outside a link that a press fills.
 */
const PRESSED = '.vp-doc a:not([class]):active, .vp-doc :not(a, pre) > code:active'

describe('a press', () => {
  it.for(['cross', 'flare', 'ink', 'turn', 'wait'])('times the fill’s %s by the press’s own', step => {
    expect(declared(`--th-fill-${step}`, PRESSED, 'fill.css')).toBe(`var(--th-fill-press-${step})`)
  })

  it.for(['cool', 'fade'])('leaves the fill’s %s, which runs once the press lifts, at its own', step => {
    expect(declared(`--th-fill-${step}`, PRESSED, 'fill.css')).toBe('')
  })

  it('times no other rule by the press’s own, so hover and keyboard focus keep the fill’s', () => {
    const pressed = styled('fill.css').filter(rule => rule.style.cssText.includes('--th-fill-press-'))
    expect(pressed.map(rule => rule.selectorText.replaceAll(/\s+/gu, ' '))).toEqual([PRESSED])
  })
})
