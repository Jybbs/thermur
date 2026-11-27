// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { Tones }                from '../../../lib/canvas/tones.ts'
import source                   from '../../../lib/rendered/identity.json' with { type: 'json' }
import { flight }               from '../../../lib/roost/flight.ts'
import { Player }               from '../../../lib/roost/player.ts'
import { READINGS, Strip }      from '../../../lib/roost/strip.ts'
import { browser }              from '../../common/browser.ts'
import { geometry }             from '../../common/front.ts'

/**
 * Plays the flight 8 s into itself and paints reading `index` onto a
 * stand-in context at the width given.
 */
function painted(index: number, width = 300) {
  const stand  = browser()
  const player = new Player(flight())
  player.step(8)
  new Strip(player, READINGS[index]!, new Tones(source)).paint(stand.pen, geometry({ h: 60, w: width }))
  return { player, stand }
}

describe('READINGS', () => {
  it('writes each value in its own unit', () => {
    expect(READINGS.map(({ write }) => write(12.345))).toEqual(['12.35', '12.3 m', '12 m'])
  })

  it('reads each row of the flight the player holds', () => {
    const player = new Player(flight())
    expect(READINGS.map(({ of }) => of(player))).toEqual([
      player.flight.polarization, player.flight.spacing, player.flight.height
    ])
  })
})

describe('Strip', () => {
  it('draws the polarization against a dashed line', () => {
    const { stand } = painted(0)
    expect(stand.pen.setLineDash).toHaveBeenCalledWith([4, 4])
  })

  it('draws the spacing against a band', () => {
    const { stand } = painted(1)
    expect(stand.pen.fillRect).toHaveBeenCalled()
    expect(stand.pen.setLineDash).not.toHaveBeenCalled()
  })

  it('draws the height inside the lowest and highest drone', () => {
    const { stand } = painted(2)
    expect(stand.pen.fill).toHaveBeenCalledTimes(2)
  })

  it('draws a narrow strip from fewer points than the frames it holds', () => {
    const { stand } = painted(0, 40)
    expect(vi.mocked(stand.pen.lineTo).mock.calls.length).toBeLessThan(120)
  })
})
