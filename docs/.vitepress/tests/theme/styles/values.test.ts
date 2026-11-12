import { describe, expect, it } from 'vitest'

import { declared } from '../../common/sheets.ts'

/**
 * The rule setting the label of the fold holding every value as a table.
 */
const LABEL = '.every-value > summary'

/**
 * The setting every control's label takes in the mono face.
 */
const CONTROL = {
  'font-weight'    : '500',
  'letter-spacing' : '0.1em',
  'text-transform' : 'uppercase'
} as const

describe('the fold’s label', () => {
  it.for(Object.entries(CONTROL))('sets %s as every control’s label does', ([property, value]) => {
    expect(declared(property, LABEL, 'values.css')).toBe(value)
  })
})
