import { describe, expect, it } from 'vitest'

import { Replayed } from '../../../lib/lookout/replayed.ts'

/**
 * Three frames half a second apart of one drone flying 10 meters east a frame,
 * its velocity's east rising by one a frame.
 */
const WATCHED = {
  drones     : 1,
  frame_s    : 0.5,
  positions  : [0, 0, 100, 10, 0, 100, 20, 0, 100],
  velocities : [1, 0, 0, 2, 0, 0, 3, 0, 0]
}

describe('Replayed', () => {
  it('starts on the first frame, kept one frame after the launch', () => {
    const flight = new Replayed(WATCHED)
    expect([...flight.positions]).toEqual([0, 0, 100])
    expect(flight.seconds).toBe(0.5)
    expect(flight.count).toBe(1)
  })

  it('plays between two frames along a straight line', () => {
    const flight = new Replayed(WATCHED)
    flight.play(0.75)
    expect([...flight.positions]).toEqual([15, 0, 100])
    expect([...flight.velocities]).toEqual([2.5, 0, 0])
    expect(flight.seconds).toBe(1.25)
  })

  it('holds on its last frame once it ends', () => {
    const flight = new Replayed(WATCHED)
    flight.play(5)
    expect([...flight.positions]).toEqual([20, 0, 100])
    expect(flight.seconds).toBe(1.5)
  })

  it('holds a flight of one frame on that frame', () => {
    const flight = new Replayed({ ...WATCHED, positions: [4, 5, 6], velocities: [1, 1, 1] })
    flight.play(1)
    expect([...flight.positions]).toEqual([4, 5, 6])
  })
})
