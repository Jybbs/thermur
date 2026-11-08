// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import * as chart from '../../../lib/rules/chart.ts'

describe('the chart’s primitives', () => {
  it('maps one span onto another along a straight line, past either end as well', () => {
    const across = chart.linear([10, 30], [100, 300])
    expect([across(10), across(20), across(30), across(40)]).toEqual([100, 200, 300, 400])
  })

  it('names each series by the custom property the identity’s stylesheet sets, the fire’s first', () => {
    expect([chart.series(0), chart.series(2)]).toEqual(['var(--th-series-1)', 'var(--th-series-3)'])
  })

  it('sets a color, a face, and an opacity as style properties and every other attribute as one', () => {
    const node = chart.svg('circle', { 'cx': 4, 'fill': chart.TONE.ink, 'opacity': 0.5, 'r': undefined })
    expect(node.namespaceURI).toBe('http://www.w3.org/2000/svg')
    expect(node.getAttribute('cx')).toBe('4')
    expect(node.hasAttribute('r')).toBe(false)
    expect(node.hasAttribute('fill')).toBe(false)
    expect(node.style.getPropertyValue('fill')).toBe('var(--vp-c-text-1)')
    expect(node.style.getPropertyValue('opacity')).toBe('0.5')
  })

  it('appends the kids it is given in order', () => {
    const node = chart.svg('g', {}, chart.svg('rect'), 'words')
    expect(node.childNodes).toHaveLength(2)
    expect(node.firstElementChild?.tagName).toBe('rect')
    expect(node.textContent).toBe('words')
  })

  it('sets words in the ink unless told another fill, at the face and size given', () => {
    const plain = chart.text(chart.FACE.mono, 12, '0.93', 5, 6)
    const muted = chart.text(chart.FACE.body, 14, 'Metric', 0, 0, { fill: chart.TONE.muted })
    expect(plain.textContent).toBe('0.93')
    expect(plain.getAttribute('font-size')).toBe('12')
    expect(plain.style.getPropertyValue('font-family')).toBe('var(--vp-font-family-mono)')
    expect(plain.style.getPropertyValue('fill')).toBe('var(--vp-c-text-1)')
    expect(muted.style.getPropertyValue('fill')).toBe('var(--vp-c-text-2)')
  })
})
