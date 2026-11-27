// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { Tones }             from '../../../lib/canvas/tones.ts'
import { Lookout, type Readout, VIEW } from '../../../lib/lookout/drawing.ts'
import type { Playing }      from '../../../lib/lookout/flight.ts'
import { geometry }          from '../../common/front.ts'

/**
 * A flight of two drones standing still at `positions`, its clock moved on by
 * `play`.
 */
function standing(positions: number[]): Playing {
  const flight = {
    count      : positions.length / 3,
    play       : (dt: number) => {
      flight.seconds += dt
    },
    positions,
    seconds    : 0,
    velocities : positions.map((_, at) => Number(at % 3 === 0))
  }
  return flight
}

/**
 * Builds the boxes a Lookout reads into, each an element of its own.
 */
const boxes = (): Readout => ({
  drones       : document.createElement('span'),
  groups       : document.createElement('span'),
  polarization : document.createElement('span'),
  time         : document.createElement('span')
})

/**
 * Reads what `readout` holds, by name.
 */
const read = (readout: Readout): Record<string, string | null> =>
  Object.fromEntries(Object.entries(readout).map(([name, box]) => [name, box.textContent]))

/**
 * Paints `lookout` over a drawing 400 by 200 pixels, returning the middle of
 * each dot it drew.
 */
function painted(lookout: Lookout): [number, number][] {
  const dots: [number, number][] = []
  const ctx = {
    fillRect: vi.fn<(x: number, y: number) => void>((x, y) => {
      dots.push([x + VIEW.dot / 2, y + VIEW.dot / 2])
    })
  }
  lookout.paint(ctx as unknown as CanvasRenderingContext2D, geometry({ h: 200, w: 400 }))
  return dots
}

describe('Lookout', () => {
  it('reads the drone count, the time, the polarization, and the groups on its fit', () => {
    const readout = boxes()
    new Lookout(standing([0, 0, 100, 10, 0, 100, 500, 0, 100]), readout, new Tones()).fit({ h: 200, w: 400 })
    expect(read(readout)).toEqual({ drones: '3', groups: '2', polarization: '1.00', time: '0.0 s' })
  })

  it('draws every drone as one dot, east across and height up, the flock’s middle at the middle', () => {
    const lookout = new Lookout(standing([-10, 0, 90, 10, 0, 110]), boxes(), new Tones())
    lookout.fit({ h: 200, w: 400 })
    const [low, high] = painted(lookout)
    expect(low![0] + high![0]).toBeCloseTo(400, 9)
    expect(low![1] + high![1]).toBeCloseTo(200, 9)
    expect(low![0]).toBeLessThan(high![0])
    expect(low![1]).toBeGreaterThan(high![1])
  })

  it('reads again only once the seconds between two readings pass', () => {
    const readout = boxes()
    const lookout = new Lookout(standing([0, 0, 100]), readout, new Tones())
    const g       = geometry({ h: 200, w: 400 })
    lookout.fit(g)
    lookout.step(VIEW.read / 2, g)
    expect(readout.time.textContent).toBe('0.1 s')
    lookout.step(VIEW.read / 4, g)
    expect(readout.time.textContent).toBe('0.1 s')
    lookout.step(VIEW.read, g)
    expect(readout.time.textContent).toBe('0.4 s')
  })

  it('glides its view toward a flock that moved rather than jumping to it', () => {
    const east    = [0, 0, 100, 10, 0, 100]
    const flight  = standing(east)
    const lookout = new Lookout(flight, boxes(), new Tones())
    lookout.fit({ h: 200, w: 400 })
    const before  = painted(lookout)[0]![0]
    east[0] = 100
    east[3] = 110
    lookout.step(0.1, geometry({ h: 200, w: 400 }))
    const after   = painted(lookout)[0]![0]
    expect(after).toBeGreaterThan(before)
    lookout.play(flight, { h: 200, w: 400 })
    expect(painted(lookout)[0]![0]).toBeCloseTo(before, 9)
  })
})
