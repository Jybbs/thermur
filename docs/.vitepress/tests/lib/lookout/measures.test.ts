import { describe, expect, it } from 'vitest'

import { Measures } from '../../../lib/lookout/measures.ts'

/**
 * Lays out drones each at one of `east` meters east, north and height zero.
 */
const along = (...east: number[]): number[] => east.flatMap(x => [x, 0, 0])

describe('Measures.groups', () => {
  it.for([
    { east: [0, 10, 20, 30], groups: 1, name: 'a chain of hops within the reach' },
    { east: [0, 45, 90], groups: 1, name: 'hops exactly the reach long' },
    { east: [0, 46, 92], groups: 3, name: 'hops past the reach' },
    { east: [300, 0, 310, 5], groups: 2, name: 'two groups listed out of order' },
    { east: [0, 0, 200], groups: 2, name: 'two drones sharing one place' }
  ])('counts $groups groups across $name', ({ east, groups }) => {
    expect(new Measures(east.length, 45).groups(along(...east))).toBe(groups)
  })

  it('joins two drones by their distance in all three axes, not by their east alone', () => {
    expect(new Measures(2, 45).groups([0, 0, 0, 10, 50, 0])).toBe(2)
    expect(new Measures(2, 45).groups([0, 0, 0, 10, 40, 0])).toBe(1)
  })

  it('reads a flock afresh on every call', () => {
    const measures = new Measures(3, 45)
    expect(measures.groups(along(0, 100, 200))).toBe(3)
    expect(measures.groups(along(0, 10, 20))).toBe(1)
  })
})

describe('Measures.polarization', () => {
  it.for([
    { name: 'a flock flying as one', polarization: 1, velocities: [10, 0, 0, 3, 0, 0] },
    { name: 'headings that cancel', polarization: 0, velocities: [10, 0, 0, -4, 0, 0] },
    { name: 'headings at right angles', polarization: Math.SQRT1_2, velocities: [0, 9, 0, 0, 0, 2] }
  ])('reads $polarization for $name, whatever each speed', ({ polarization, velocities }) => {
    expect(new Measures(2, 45).polarization(velocities)).toBeCloseTo(polarization, 9)
  })
})
