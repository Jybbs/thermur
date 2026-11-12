import fs from 'node:fs'

import { describe, expect, it } from 'vitest'

import { styled } from '../../common/sheets.ts'

/**
 * The sheets the identity's widgets are drawn in.
 */
const SHEETS = ['faces', 'heat', 'marks', 'palette', 'specimen'].map(name => `identity/${name}.css`)

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
