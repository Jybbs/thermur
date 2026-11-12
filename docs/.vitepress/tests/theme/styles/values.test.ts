import fs from 'node:fs'

import { describe, expect, it } from 'vitest'

import { fontStack }                  from '../../../lib/identity.ts'
import source                         from '../../../lib/rendered/identity.json' with { type: 'json' }
import { computed, declared, styled } from '../../common/sheets.ts'

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
 * Every sheet under the theme's styles, those in a family's folder included.
 */
const SHEETS = fs.readdirSync(new URL('../../../theme/styles/', import.meta.url), { recursive: true })
  .map(String)
  .filter(name => name.endsWith('.css'))

/**
 * The mono uppercase rules that set a readout's value rather than a control's
 * label.
 */
const VALUES = new Set(['.rules-chip-value'])

describe('the fold’s label', () => {
  it('sets the identity’s mono face at 500, as every control’s label does', () => {
    expect(computed(FOLD, 'font-family', '#label', 'values.css')).toBe(fontStack(source.faces.mono).replaceAll('\'', '"'))
    expect(computed(FOLD, 'font-weight', '#label', 'values.css')).toBe('500')
  })

  it.for([['letter-spacing', '0.1em'], ['text-transform', 'uppercase']] as const)('sets %s as every control’s label does', ([property, value]) => {
    expect(declared(property, LABEL, 'values.css')).toBe(value)
  })
})

describe('every control’s label', () => {
  it('spaces each mono uppercase label in every sheet at 0.1em', () => {
    const labels = SHEETS.flatMap(sheet => styled(sheet)
      .filter(rule => rule.style.cssText.includes('var(--vp-font-family-mono)'))
      .filter(rule => rule.style.getPropertyValue('text-transform') === 'uppercase')
      .filter(rule => !VALUES.has(rule.selectorText))
      .map(rule => [sheet, rule.selectorText.replaceAll(/\s+/gu, ' '), rule.style.getPropertyValue('letter-spacing')]))
    expect(labels).toContainEqual(['values.css', LABEL, '0.1em'])
    expect(labels.filter(([, , spacing]) => spacing !== '0.1em')).toEqual([])
  })
})
