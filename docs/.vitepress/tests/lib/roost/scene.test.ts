// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { Marks }                       from '../../../lib/canvas/marks.ts'
import { Tones }                       from '../../../lib/canvas/tones.ts'
import { flight }                      from '../../../lib/roost/flight.ts'
import { Player }                      from '../../../lib/roost/player.ts'
import { type Camera, type Shot, Scene } from '../../../lib/roost/scene.ts'
import { SHOT }                        from '../../../lib/roost/shot.ts'
import { browser }                     from '../../common/browser.ts'
import { geometry }                    from '../../common/front.ts'

/**
 * Builds a scene over the page's flight, playing `seconds` into the loop,
 * that aims by `shot`, with the flock drawn from three one-point birds.
 */
function scene(shot: Shot, seconds = 8) {
  const stand  = browser()
  const tones  = new Tones()
  const player = new Player(flight())
  const shown  = vi.fn<() => void>()
  player.step(seconds)
  return { player, scene: new Scene(new Marks(['M0 0Z', 'M1 1Z', 'M2 2Z'], tones), tones, player, shot, shown), shown, stand }
}

const flat: Shot = {
  aim   : (camera: Camera) => Object.assign(camera, { across: 600, away: Infinity, down: 400, pitch: Math.PI / 2, scale: 0.5, x: 0, y: 0 }),
  bird  : [10, 4, 18],
  drops : false,
  grid  : 100,
  trail : 6
}

describe('paint', () => {
  it('draws the roost, its trails, and each of the 200 drones, then says the moment shown', () => {
    const { scene: drawn, shown, stand } = scene(SHOT)
    drawn.paint(stand.pen, geometry(), 1)
    expect(shown).toHaveBeenCalledOnce()
    expect(stand.pen.fill).toHaveBeenCalled()
    expect(stand.pen.arc).toHaveBeenCalledWith(expect.any(Number), expect.any(Number), 16, 0, Math.PI * 2)
  })

  it('draws a view with no perspective, no drops, no ring, and no burn', () => {
    const { scene: drawn, stand } = scene(flat)
    drawn.paint(stand.pen, geometry({ scheme: 'light' }), 2)
    expect(stand.pen.arc).not.toHaveBeenCalled()
    expect(stand.pen.fillRect).not.toHaveBeenCalled()
  })

  it('draws no trail before the flight has flown a frame', () => {
    const { scene: drawn, stand } = scene(flat, 0)
    drawn.paint(stand.pen, geometry(), 1)
    expect(stand.pen.stroke).toHaveBeenCalled()
  })
})

describe('circle', () => {
  it('traces a closed path around the roost', () => {
    const { scene: drawn, stand } = scene(flat)
    drawn.paint(stand.pen, geometry(), 1)
    vi.mocked(stand.pen.lineTo).mockClear()
    drawn.circle(stand.pen, 100)
    expect(stand.pen.lineTo).toHaveBeenCalledTimes(120)
  })
})

describe('step', () => {
  it('carries the player on', () => {
    const { player, scene: drawn } = scene(flat, 0)
    drawn.step(2)
    expect(player.lift).toBeLessThan(1)
  })
})
