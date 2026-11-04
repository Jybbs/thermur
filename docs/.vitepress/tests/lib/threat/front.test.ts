import { describe, expect, it } from 'vitest'

import { Front } from '../../../lib/threat/front.ts'

const front = new Front()

describe('the model of the threat', () => {
  it('reads the record’s two minutes of snapshots, the lowest level’s center, and the far column', () => {
    expect([front.interval, front.lowest, front.snapshots, front.spacing, front.span, front.far])
      .toEqual([15, 10, 9, 40, 120, 3980])
  })

  it('advances the front one meter each second from the western edge', () => {
    expect([front.at(0), front.at(60), front.at(120)]).toEqual([0, 60, 120])
  })

  it('places each cell’s center half a spacing into its column', () => {
    expect([front.center(0), front.center(1)]).toEqual([20, 60])
  })

  it('carries half the anomaly at the front, nearly all behind it, and nearly none ahead', () => {
    expect(front.profile(60, 60)).toBe(0.5)
    expect(front.profile(60, -200)).toBeCloseTo(1, 4)
    expect(front.profile(60, 320)).toBeCloseTo(0, 4)
  })

  it.for([
    { height: 10, kelvin: 317.5, name: 'warmed behind the front', seconds: 60, x: 20 },
    { height: 590, kelvin: 294.2, name: 'cooled with height ahead of the front', seconds: 60, x: 3980 }
  ])('reads the air $name as the package’s dataset holds it', ({ height, kelvin, seconds, x }) => {
    expect(front.air(height, seconds, x)).toBeCloseTo(kelvin, 1)
  })
})
