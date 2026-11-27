// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import { browser }       from '../../common/browser.ts'
import { PLAN, flight }  from '../../../lib/roost/flight.ts'
import { Player }        from '../../../lib/roost/player.ts'
import type { Camera }   from '../../../lib/roost/scene.ts'
import { SHOT, noteOf }  from '../../../lib/roost/shot.ts'

const camera = (): Camera => ({ across: 0, away: 0, down: 0, facing: 0, pitch: 0, scale: 1, x: 0, y: 0 })

describe('SHOT.aim', () => {
  it('follows the ringed drone, facing west with the view tilted up from the ground, once panned in', () => {
    browser()
    const player = new Player(flight())
    player.step(4)
    const aimed = camera()
    SHOT.aim(aimed, player, 1200, 700)
    expect(aimed.facing).toBeCloseTo(-Math.PI / 2, 6)
    expect(aimed.pitch).toBeCloseTo(38 * Math.PI / 180, 6)
    expect(aimed.across).toBe(600)
    expect(aimed.x).toBeCloseTo(player.read(player.flight.followed, 2, 0), 6)
  })

  it('lifts toward the whole roost as the pan begins, standing between the two views', () => {
    browser()
    const player = new Player(flight())
    player.step(0)
    const whole = camera()
    SHOT.aim(whole, player, 1200, 700)
    expect(whole.pitch).toBeCloseTo(34 * Math.PI / 180, 6)
    expect(whole.x).toBeCloseTo(player.flight.roost.center_m[0], 6)
    player.step(1.5)
    const between = camera()
    SHOT.aim(between, player, 1200, 700)
    expect(between.pitch).toBeGreaterThan(34 * Math.PI / 180)
    expect(between.pitch).toBeLessThan(38 * Math.PI / 180)
  })
})

describe('noteOf', () => {
  it('names the drones, the roost, the fire, the pace, and what the bands say', () => {
    const note = noteOf(flight())
    expect(note).toContain('200 drones over a roost 1 km across')
    expect(note).toContain(`${PLAN.seconds} s shown ${PLAN.pace} times as fast`)
    expect(note).toContain(SHOT.burn!.said)
  })
})
