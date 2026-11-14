// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { RELEASED, box, browser } from '../../../.vitepress/tests/common/browser.ts'
import * as markup                from '../../src/drawing/markup.ts'

describe('the helpers that write a drawing’s markup', () => {
  it('sets a color, a face, and an opacity as style properties and every other attribute as one', () => {
    const node = markup.svg('circle', { 'cx': 4, 'fill': markup.TONE.ink, 'opacity': 0.5, 'r': undefined })
    expect(node.namespaceURI).toBe('http://www.w3.org/2000/svg')
    expect(node.getAttribute('cx')).toBe('4')
    expect(node.hasAttribute('r')).toBe(false)
    expect(node.hasAttribute('fill')).toBe(false)
    expect(node.style.getPropertyValue('fill')).toBe('var(--vp-c-text-1)')
    expect(node.style.getPropertyValue('opacity')).toBe('0.5')
  })

  it('appends the kids it is given in order', () => {
    const node = markup.svg('g', {}, markup.svg('rect'), 'words')
    expect(node.childNodes).toHaveLength(2)
    expect(node.firstElementChild?.tagName).toBe('rect')
    expect(node.textContent).toBe('words')
  })

  it('sets words in the ink unless told another fill, at the face and size given', () => {
    const plain = markup.text(markup.FACE.mono, 12, '0.93', 5, 6)
    const muted = markup.text(markup.FACE.body, 14, 'Metric', 0, 0, { fill: markup.TONE.muted })
    expect(plain.textContent).toBe('0.93')
    expect(plain.getAttribute('font-size')).toBe('12')
    expect(plain.style.getPropertyValue('font-family')).toBe('var(--vp-font-family-mono)')
    expect(plain.style.getPropertyValue('fill')).toBe('var(--vp-c-text-1)')
    expect(muted.style.getPropertyValue('fill')).toBe('var(--vp-c-text-2)')
  })

  it('calls back only once the box takes a nonzero width other than the one last drawn', () => {
    const stand   = browser()
    const host    = box('rules-host')
    const widths  = [0, 640, 640.3, 390]
    const changed = vi.fn<(width: number) => void>()
    vi.spyOn(host, 'getBoundingClientRect').mockImplementation(() => ({ width: widths.shift() }) as DOMRect)
    const release = markup.widthOf(host, changed)
    for (let k = 0; k < 4; k++) stand.reflow(host)
    expect(changed.mock.calls).toEqual([[640], [390]])
    release()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('places a control over a drawing in pixels, leaving an unset size to its stylesheet', () => {
    expect(markup.placed({ left: 12, top: 3.5 })).toEqual({ height: undefined, left: '12px', top: '3.5px', width: undefined })
    expect(markup.placed({ height: 28, left: 0, top: 10, width: 200 })).toMatchObject({ height: '28px', width: '200px' })
  })
})
