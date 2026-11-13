import { describe, expect, it } from 'vitest'

import { Front }    from '../../../lib/threat/front.ts'
import { Playback } from '../../../lib/threat/playback.ts'

/**
 * Runs `playback` on by `seconds` of the reader's time in steps of a tenth of a
 * second.
 */
function ran(playback: Playback, seconds: number): Playback {
  for (let elapsed = 0; elapsed < seconds - 1e-9; elapsed += 0.1) playback.step(0.1)
  return playback
}

const run = (): Playback => new Playback(new Front())

/**
 * Runs a clock on until its second run has just begun.
 */
function restarted(): Playback {
  const playback = run()
  while (playback.fading === 0) playback.step(0.1)
  return playback
}

describe('the clock on its own', () => {
  it('runs the threat’s two minutes at ten times their pace', () => {
    expect(ran(run(), 6).seconds).toBeCloseTo(60)
    expect(ran(run(), 12).seconds).toBeCloseTo(120)
  })

  it('rests at the end of a run, the time held', () => {
    expect(ran(run(), 13.4).seconds).toBe(120)
    expect(ran(run(), 13.4).fading).toBe(0)
  })

  it('enters the next front at the western edge as the last run fades, never running backward', () => {
    const playback = restarted()
    expect([playback.seconds, playback.fading]).toEqual([0, 1])
    let last = playback.seconds
    for (let step = 0; step < 30; step++) {
      playback.step(0.1)
      expect(playback.seconds).toBeGreaterThanOrEqual(last)
      last = playback.seconds
    }
    expect(playback.fading).toBe(0)
  })
})

describe('a reader’s move on the rail', () => {
  it('glides to the time set rather than jumping', () => {
    const playback = run()
    playback.seek(60)
    playback.step(0.016)
    expect(playback.seconds).toBeGreaterThan(0)
    expect(playback.seconds).toBeLessThan(60)
    expect(ran(playback, 2).seconds).toBeCloseTo(60, 2)
  })

  it('holds the run until a few seconds after the move, then carries on from the time set', () => {
    const playback = run()
    playback.seek(60)
    expect(ran(playback, 3.9).target).toBe(60)
    expect(ran(playback, 0.3).target).toBeGreaterThan(60)
  })

  it('holds the run for as long as the rail is pressed, however long, and swells the grip', () => {
    const playback = run()
    playback.press()
    playback.seek(30)
    expect(ran(playback, 20).target).toBe(30)
    expect(playback.grip).toBeCloseTo(1, 2)
    playback.release()
    expect(ran(playback, 3.9).target).toBe(30)
    expect(playback.grip).toBeCloseTo(0, 2)
    expect(ran(playback, 0.3).target).toBeGreaterThan(30)
  })

  it.for([[-10, 0], [500, 120]])('holds a time of %i inside the run, at %i', ([seconds, held]) => {
    const playback = run()
    playback.seek(seconds!)
    expect(playback.target).toBe(held)
  })

  it('clears the fading of the run before', () => {
    const playback = restarted()
    playback.seek(30)
    expect(playback.fading).toBe(0)
  })
})

describe('a drawing standing still', () => {
  it('shows the middle of a run that has not moved', () => {
    const playback = run()
    playback.settle()
    expect([playback.seconds, playback.target]).toEqual([60, 60])
  })

  it('takes a time a reader set at once', () => {
    const playback = run()
    playback.seek(90)
    playback.settle()
    expect(playback.seconds).toBe(90)
  })

  it('stands where a paused run stopped', () => {
    const playback = ran(run(), 3)
    playback.settle()
    expect(playback.seconds).toBeCloseTo(30)
  })

  it('shows the grip swollen under a press, at once', () => {
    const playback = run()
    playback.press()
    playback.settle()
    expect(playback.grip).toBe(1)
  })

  it('shows the grip at rest once the press lets go, at once', () => {
    const playback = run()
    playback.press()
    playback.settle()
    playback.release()
    playback.settle()
    expect(playback.grip).toBe(0)
  })
})
