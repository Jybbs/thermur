// @vitest-environment happy-dom
import { enableAutoUnmount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import RulesReach  from '../../../theme/rules/reach.ts'
import { charted } from '../../common/rules.ts'

enableAutoUnmount(afterEach)

describe('the names on the heading chart', () => {
  it('sets the overdamped limit in the ink at regular weight and the default in the fire’s hue in bold', () => {
    const chart = charted(RulesReach, 1024)
    const named = (line: string): [string, string | null] => {
      const text = [...chart.host.querySelectorAll('text')].find(node => node.textContent === line)!
      return [text.style.getPropertyValue('fill'), text.getAttribute('font-weight')]
    }
    expect(named('Overdamped limit')).toEqual(['var(--vp-c-text-1)', '400'])
    expect(named('Consensus, the default')).toEqual(['var(--th-series-1)', '600'])
    expect(named('Inertial spin')).toEqual(['var(--th-series-2)', '400'])
  })
})
