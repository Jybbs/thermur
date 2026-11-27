// @vitest-environment happy-dom
import { enableAutoUnmount }                   from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Tones }      from '../../../lib/canvas/tones.ts'
import { ALONG }      from '../../../lib/rules/falcon.ts'
import RulesNeighbors from '../../../theme/rules/neighbors.ts'
import { charted }    from '../../common/rules.ts'

enableAutoUnmount(afterEach)
afterEach(() => {
  document.documentElement.classList.remove('dark')
  vi.restoreAllMocks()
})

describe('the farthest ranked chip', () => {
  it.each([
    ['dark', new Tones().flock('dark', ALONG.seven, 1)],
    ['light', new Tones().ink('light')]
  ] as const)('writes its name under the %s scheme in %s', (scheme, color) => {
    document.documentElement.classList.toggle('dark', scheme === 'dark')
    const chart = charted(RulesNeighbors, 672, true)
    chart.stand.see(true)
    expect(chart.host.querySelector<HTMLElement>('.kit-chip-name')!.style.color).toBe(color)
  })
})
