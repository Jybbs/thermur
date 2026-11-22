import { describe, expect, it } from 'vitest'

import palette                        from '../../../theme/styles/identity/palette.css?raw'
import { computed, declared, styled } from '../../common/sheets.ts'

/**
 * A chart series' switch, turned off.
 */
const OFF = '<button class="identity-switch" role="switch" aria-checked="false">'
          + '<span class="identity-caps">Chart 1</span><svg class="identity-spark"></svg></button>'
/**
 * The text of each sheet the identity's widgets are drawn in, by its name.
 */
const SHEETS = import.meta.glob<string>('./{faces,heat,marks,palette,specimen}.css', {
  base   : '../../../theme/styles/identity/',
  eager  : true,
  import : 'default',
  query  : '?raw'
})

/**
 * Reads the stroke width the palette's sheet sets for part `part` of a level.
 */
const widthOf = (part: string): number =>
  Number(declared('stroke-width', `.identity-level-${part}`, palette))

describe.for(Object.keys(SHEETS))('%s', name => {
  const sheet = SHEETS[name]!

  it('closes every block it opens', () => {
    expect(sheet.split('{').length).toBe(sheet.split('}').length)
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
  expect(computed(OFF, 'opacity', '.identity-switch', palette)).toBe('')
  expect(computed(OFF, 'opacity', '.identity-spark', palette)).toBe('0.45')
})

it('outlines each level line wider than the stroke drawn over it', () => {
  expect(widthOf('edge')).toBeGreaterThan(widthOf('line') + 1)
})
