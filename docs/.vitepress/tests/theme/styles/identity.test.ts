import fs from 'node:fs'

import { describe, expect, it } from 'vitest'

import { computed, declared, styled } from '../../common/sheets.ts'

/**
 * A chart series' switch, turned off.
 */
const OFF = '<button class="identity-switch" role="switch" aria-checked="false">'
          + '<span class="identity-caps">Chart 1</span><svg class="identity-spark"></svg></button>'
/**
 * The sheets the identity's widgets are drawn in.
 */
const SHEETS = ['faces', 'heat', 'marks', 'palette', 'specimen'].map(name => `identity/${name}.css`)

/**
 * Reads the stroke width the palette's sheet sets for part `part` of a level.
 */
const widthOf = (part: string): number =>
  Number(declared('stroke-width', `.identity-level-${part}`, 'identity/palette.css'))

describe.for(SHEETS)('%s', sheet => {
  const text = fs.readFileSync(new URL(`../../../theme/styles/${sheet}`, import.meta.url), 'utf8')

  it('closes every block it opens', () => {
    expect(text.split('{').length).toBe(text.split('}').length)
  })

  it('times every transition and animation by the site’s settle time', () => {
    const timed = styled(sheet).flatMap(rule => [rule.style.transition, rule.style.animation].filter(Boolean))
    expect(timed.filter(timing => !/var\(--th-settle\)/u.test(timing))).toEqual([])
  })

  it('sets no background on a link or a chip the fill draws on', () => {
    const drawn = styled(sheet).filter(rule => /\ba(?![\w-])|\bcode\b/u.test(rule.selectorText))
    expect(drawn.map(rule => rule.style.background)).toEqual(drawn.map(() => ''))
  })
})

it('dims only the spark of a series switched off, leaving its label at full strength', () => {
  expect(computed(OFF, 'opacity', '.identity-switch', 'identity/palette.css')).toBe('')
  expect(computed(OFF, 'opacity', '.identity-spark', 'identity/palette.css')).toBe('0.45')
})

it('outlines each level line wider than the stroke drawn over it', () => {
  expect(widthOf('edge')).toBeGreaterThan(widthOf('line') + 1)
})
