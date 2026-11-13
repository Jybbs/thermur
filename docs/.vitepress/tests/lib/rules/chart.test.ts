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
})
