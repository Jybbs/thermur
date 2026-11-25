// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import credit           from '../../../lib/rendered/credit.json' with { type: 'json' }
import CreditsSnapshots from '../../../theme/credits/snapshots.ts'

enableAutoUnmount(afterEach)

describe('CreditsSnapshots', () => {
  it('draws a dot for each snapshot the record’s sample holds, evenly along one line', () => {
    const centers = mount(CreditsSnapshots).findAll('circle').map(dot => Number(dot.attributes('cx')))
    const steps   = centers.slice(1).map((center, index) => center - centers[index]!)
    expect(centers).toHaveLength(credit.snapshots)
    expect(new Set(steps).size).toBe(1)
  })

  it('warms every dot through the one gradient, from cool at the first time to hot at the last', () => {
    const band = mount(CreditsSnapshots)
    const ramp = band.get('linearGradient')
    expect(ramp.attributes()).toMatchObject({ gradientUnits: 'userSpaceOnUse', x1: '0' })
    expect(band.attributes('viewBox')!.split(' ')[2]).toBe(ramp.attributes('x2'))
    expect(ramp.findAll('stop').map(stop => stop.attributes('class')))
      .toEqual(['credits-cool', 'credits-mid', 'credits-hot'])
    expect(new Set(band.findAll('circle').map(dot => dot.attributes('fill'))))
      .toEqual(new Set([`url(#${ramp.attributes('id')!})`]))
  })

  it('hides from a screen reader, which reads the line’s own times', () => {
    expect(mount(CreditsSnapshots).attributes('aria-hidden')).toBe('true')
  })
})
