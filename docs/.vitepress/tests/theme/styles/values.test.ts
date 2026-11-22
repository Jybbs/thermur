import { describe, expect, it } from 'vitest'

import { fontStack }                                 from '../../../lib/identity.ts'
import source                                        from '../../../lib/rendered/identity.json' with { type: 'json' }
import values                                        from '../../../theme/styles/values.css?raw'
import { CONTROL_LABEL, computed, declared, styled } from '../../common/sheets.ts'

/**
 * The rule setting the label of the fold holding every value as a table.
 */
const LABEL = '.every-value > summary'

/**
 * A fold beneath a widget, inside a page that sets the mono face the way the
 * identity's stylesheet does.
 */
const FOLD = `<div style="--vp-font-family-mono: ${fontStack(source.faces.mono)}">`
  + '<details class="every-value"><summary id="label">Every value as a table</summary></details></div>'

/**
 * The text of every sheet under the theme's styles, those in a family's folder
 * included, by its path there.
 */
const SHEETS = import.meta.glob<string>('./**/*.css', {
  base   : '../../../theme/styles/',
  eager  : true,
  import : 'default',
  query  : '?raw'
})

/**
 * The mono uppercase rules that set a readout's value rather than a control's
 * label.
 */
const VALUES = new Set(['.rules-chip-value'])

describe('the fold’s label', () => {
  it('sets the identity’s mono face at 500, as every control’s label does', () => {
    expect(computed(FOLD, 'font-family', '#label', values)).toBe(fontStack(source.faces.mono).replaceAll('\'', '"'))
    expect(computed(FOLD, 'font-weight', '#label', values)).toBe(CONTROL_LABEL['font-weight'])
  })

  it.for(['letter-spacing', 'text-transform'] as const)('sets %s as every control’s label does', property => {
    expect(declared(property, LABEL, values)).toBe(CONTROL_LABEL[property])
  })
})

describe('every control’s label', () => {
  it('spaces each mono uppercase label in every sheet at 0.1em', () => {
    const labels = Object.entries(SHEETS).flatMap(([name, sheet]) => styled(sheet)
      .filter(rule => rule.style.cssText.includes('var(--vp-font-family-mono)'))
      .filter(rule => rule.style.getPropertyValue('text-transform') === 'uppercase')
      .filter(rule => !VALUES.has(rule.selectorText))
      .map(rule => [name, rule.selectorText.replaceAll(/\s+/gu, ' '), rule.style.getPropertyValue('letter-spacing')]))
    const { 'letter-spacing': spacing } = CONTROL_LABEL
    expect(labels).toContainEqual(['./values.css', LABEL, spacing])
    expect(labels.filter(([, , each]) => each !== spacing)).toEqual([])
  })
})
