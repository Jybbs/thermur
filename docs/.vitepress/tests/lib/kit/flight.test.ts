// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import { Flight } from '../../../lib/kit/flight.ts'

/**
 * Reads where every bird of `flight` stands.
 */
const places = (flight: Flight): string[] => flight.birds.map(bird => `${bird.getAttribute('cx')},${bird.getAttribute('cy')}`)

describe('Flight', () => {
  it('draws every bird on a point of its own as it is built', () => {
    const flight = new Flight()
    expect(flight.node.querySelectorAll('circle')).toHaveLength(flight.birds.length)
    expect(new Set(places(flight)).size).toBe(flight.birds.length)
  })

  it('moves the birds already drawn as time runs, building none', () => {
    const flight = new Flight()
    const before = places(flight)
    const birds  = [...flight.birds]
    flight.draw(2)
    expect(places(flight)).not.toEqual(before)
    expect(flight.birds).toEqual(birds)
    expect(flight.node.querySelectorAll('circle')).toHaveLength(birds.length)
  })

  it('draws the gathered flock tighter than the loose one', () => {
    const flight = new Flight()
    const spread = (): number => {
      const xs = flight.birds.map(bird => Number(bird.getAttribute('cx')))
      return Math.max(...xs) - Math.min(...xs)
    }
    const loose     = spread()
    flight.gathered = 1
    flight.draw(0)
    expect(spread()).toBeLessThan(loose / 4)
  })
})
