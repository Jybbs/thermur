import { describe, expect, it } from 'vitest'

import { SEAL, groundOf, rimOf, rimPath } from '../../../lib/credits/seal.ts'
import source                             from '../../../lib/rendered/credit.json' with { type: 'json' }

/**
 * A credit of a dataset other than the one the record holds.
 */
const CREDIT = { ...source, license: 'CC0 1.0', surnames: ['Author'], year: 1999 }

describe('the rim’s words', () => {
  it('reads the creators along its top and the year and the license along its foot', () => {
    expect(rimOf(CREDIT)).toEqual({ foot: '1999 · CC0 1.0', top: 'Author' })
  })

  it('reads the record’s own credit where no other is given', () => {
    const { license, surnames, year } = source
    expect(rimOf()).toEqual({ foot: `${year} · ${license}`, top: surnames.join(' & ') })
  })
})

describe('the rim’s arcs', () => {
  it('runs the top one clockwise from the seal’s foot, so its middle is the head', () => {
    expect(rimPath(40, true)).toBe('M60 100a40 40 0 0 1 0 -80a40 40 0 0 1 0 80')
  })

  it('runs the foot one the other way from the seal’s head, so its middle is the foot', () => {
    expect(rimPath(40, false)).toBe('M60 20a40 40 0 0 0 0 80a40 40 0 0 0 0 -80')
  })

  it('stands both between the core and the hairline', () => {
    for (const radius of [SEAL.rim.foot, SEAL.rim.top]) {
      expect(radius).toBeGreaterThan(SEAL.core)
      expect(radius).toBeLessThan(SEAL.hair)
    }
  })
})

it('runs the ground from the middle to the core’s ring', () => {
  expect(groundOf() ** 2 + (SEAL.flame.y - SEAL.center) ** 2).toBeCloseTo(SEAL.core ** 2)
  expect(groundOf()).toBeGreaterThan(SEAL.flame.glow)
})
