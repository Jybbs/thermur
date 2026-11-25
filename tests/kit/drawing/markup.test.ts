// @vitest-environment happy-dom
import { type Mock, describe, expect, it, vi } from 'vitest'

import { type Browser, RELEASED, box, browser } from '../../site/common/browser.ts'
import * as markup                             from '../../../docs/kit/src/drawing/markup.ts'

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

  it('sets a caption over a control in the muted tone, in the class the kit’s label rule sets', () => {
    const caption = markup.text(markup.FACE.mono, 10.5, 'LAUNCH SPACING', 0, 0, markup.CAPTION)
    expect(caption.getAttribute('class')).toBe('kit-caption')
    expect(caption.style.getPropertyValue('fill')).toBe('var(--vp-c-text-2)')
  })

  it('places a control over a drawing in pixels, leaving an unset size to its stylesheet', () => {
    expect(markup.placed({ left: 12, top: 3.5 })).toEqual({ height: undefined, left: '12px', top: '3.5px', width: undefined })
    expect(markup.placed({ height: 28, left: 0, top: 10, width: 200 })).toMatchObject({ height: '28px', width: '200px' })
  })
})

/**
 * Watches the width of a fresh box over the stand-ins, returning the stand-ins,
 * the box, the callback the width reaches, and what releases the watch.
 */
function watched(): { changed: Mock<(width: number) => void>, host: HTMLElement, release: () => void, stand: Browser } {
  const stand   = browser()
  const host    = box('rules-host')
  const changed = vi.fn<(width: number) => void>()
  return { changed, host, release: markup.widthOf(host, changed), stand }
}

describe('the width a drawing is drawn at', () => {
  it('calls back only once the box takes a nonzero width other than the one last drawn', () => {
    const { changed, host, release, stand } = watched()
    for (const width of [0, 640, 640.3, 390]) {
      stand.reflow(host, width)
      stand.frames.run(1)
    }
    expect(changed.mock.calls).toEqual([[640], [390]])
    release()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('draws at a new width on the next frame, measuring nothing inside the observer’s callback', () => {
    const measured                 = vi.spyOn(Element.prototype, 'getBoundingClientRect')
    const { changed, host, stand } = watched()
    stand.reflow(host, 640)
    expect(changed).not.toHaveBeenCalled()
    stand.frames.run(1)
    expect(changed).toHaveBeenCalledExactlyOnceWith(640)
    expect(measured).not.toHaveBeenCalled()
  })

  it('draws once at the last width a frame sees, dropping the frame on release', () => {
    const { changed, host, release, stand } = watched()
    stand.reflow(host, 640)
    stand.reflow(host, 390)
    stand.frames.run(1)
    expect(changed.mock.calls).toEqual([[390]])
    stand.reflow(host, 512)
    release()
    expect(stand.held()).toEqual(RELEASED)
    stand.frames.run(1)
    expect(changed).toHaveBeenCalledOnce()
  })

  it('draws at the width a box returns to within one frame after leaving it', () => {
    const { changed, host, stand } = watched()
    stand.reflow(host, 640)
    stand.frames.run(1)
    stand.reflow(host, 390)
    stand.reflow(host, 640)
    stand.frames.run(1)
    expect(changed.mock.calls.at(-1)).toEqual([640])
  })
})
