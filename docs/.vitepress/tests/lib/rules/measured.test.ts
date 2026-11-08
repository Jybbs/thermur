import { describe, expect, it } from 'vitest'

import * as measured from '../../../lib/rules/measured.ts'

/**
 * A rule measured at three spacings, each with its mean and standard error.
 */
const RULE: measured.Rule = {
  color  : 'var(--th-series-1)',
  label  : 'Constant, the default',
  points : [
    { error: 0.02, given: 10, mean: 0.9 },
    { error: 0.04, given: 15, mean: 0.8 },
    { error: 0.06, given: 25, mean: 0.6 }
  ],
  short  : 'Constant'
}

describe('a rule between its measured points', () => {
  it.for([
    { mean: 0.9, spacing: 10 },
    { mean: 0.85, spacing: 12.5 },
    { mean: 0.8, spacing: 15 },
    { mean: 0.7, spacing: 20 },
    { mean: 0.6, spacing: 25 }
  ])('reads $mean at $spacing m', ({ mean, spacing }) => {
    expect(measured.between(RULE, spacing).mean).toBeCloseTo(mean, 10)
    expect(measured.between(RULE, spacing).given).toBe(spacing)
  })

  it('holds the nearest end beyond the measured spacings', () => {
    expect(measured.between(RULE, 5)).toMatchObject({ error: 0.02, mean: 0.9 })
    expect(measured.between(RULE, 40)).toMatchObject({ error: 0.06, mean: 0.6 })
  })

  it('writes a point as its mean beside its standard error, to two places', () => {
    expect(measured.reading({ error: 0.0139, given: 10, mean: 0.9287 })).toBe('0.93 ± 0.01')
  })
})

describe('the rules out of a trial', () => {
  const trial: measured.Trial = {
    given    : { dense: 10, sparse: 25 },
    measured : {
      metric    : { dense: { error: 0.1, mean: 0.7 }, sparse: { error: 0.2, mean: 0.5 } },
      reference : { sparse: { error: 0.01, mean: 0.94 }, dense: { error: 0.02, mean: 0.93 } }
    }
  }

  it('lists the rules in the order the names give, each colored by its place and named twice', () => {
    const names = { reference: ['Constant, the default', 'Constant'], metric: ['Metric', 'M'] } as const
    const rules = measured.rulesOf(names, trial)
    expect(rules.map(rule => [rule.label, rule.short, rule.color])).toEqual([
      ['Constant, the default', 'Constant', 'var(--th-series-1)'],
      ['Metric', 'M', 'var(--th-series-2)']
    ])
  })

  it('sorts each rule’s points by what each condition sets', () => {
    const [reference] = measured.rulesOf({ reference: ['Constant', 'Constant'] }, trial)
    expect(reference!.points).toEqual([
      { error: 0.02, given: 10, mean: 0.93 },
      { error: 0.01, given: 25, mean: 0.94 }
    ])
  })

  it('carries every rule the page names out of the trials `mise measure` wrote', () => {
    for (const written of Object.values(measured.TRIALS)) {
      expect(Object.keys(written.measured)).toContain('reference')
      expect(Object.keys(written.given).length).toBeGreaterThan(1)
    }
  })

  it('places the coupling by launch spacing and the heading by turn, as `mise render` wrote them', () => {
    expect(measured.TRIALS.coupling.given).toEqual({ dense: 10, middle: 15, sparse: 25 })
    expect(measured.TRIALS.heading.given).toEqual({ large: 90, small: 20 })
  })

  it('reads a field only from the conditions that set it', () => {
    const changes = { large: { frame_every: 5, turn_deg: 90 }, spontaneous: {} }
    expect(measured.setting(changes, 'turn_deg')).toEqual({ large: 90 })
    expect(measured.setting(changes, 'spacing_m')).toEqual({})
  })
})
