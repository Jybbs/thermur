import { type MockInstance, vi } from 'vitest'

import type { Marks }               from '../../lib/canvas/marks.ts'
import { Kept }                     from '../../lib/crop/kept.ts'
import { menusOf }                  from '../../lib/crop/sentence.ts'
import { FlightStrip, stripY }      from '../../lib/crop/strip.ts'
import { FIRE }                     from '../../lib/front/drawing.ts'
import { Flock }                    from '../../lib/front/flock.ts'
import type { Murmur }              from '../../lib/front/sky.ts'
import type { Browser }             from './browser.ts'
import { FLOCK, geometry, skyOver } from './front.ts'

/**
 * One bird as a paint drew it: which bird of the flock, and where and how long.
 */
interface Draw {
  index  : number
  length : number
  x      : number
  y      : number
}

/**
 * A strip fitted to its box beside what a case reads it through.
 */
export interface Flown {
  bird  : MockInstance<Marks['bird']>
  birds : Flock
  kept  : Kept
  part  : FlightStrip
  stand : Browser
}

/**
 * The flock the front page flies, settled briefly enough for a case to fly many
 * times over.
 */
export const CROWD: Murmur = { count: 340, settle: 0.5 }
/**
 * The box a strip draws in, 8rem tall in a column 688 pixels wide.
 */
export const STRIP = { h: 128, w: 688 } as const

/**
 * Lists the birds a strip's paints drew, matching each to the flock by its
 * heading.
 */
export const draws = ({ bird, birds }: Pick<Flown, 'bird' | 'birds'>): Draw[] =>
  bird.mock.calls.map(([, , , x, y, angle, length]) => ({
    index : birds.vx.findIndex((vx, i) => Math.atan2(birds.vy[i]!, vx) === angle),
    length,
    x,
    y
  }))

/**
 * Fits a strip flying `murmur` to `STRIP`, with the menus then set to `picks`.
 */
export function flown(murmur: Murmur = FLOCK, picks: readonly number[] = [0, 0, 0]): Flown {
  const { marks, stand, tones } = skyOver()
  const settle                  = vi.spyOn(Flock.prototype, 'settle')
  const kept                    = new Kept()
  const part                    = new FlightStrip(kept, marks, tones, FIRE, murmur)
  part.fit(geometry(STRIP))
  for (const [index, menu] of menusOf(kept).entries()) menu.choices[picks[index]!]!.apply()
  return { bird: vi.spyOn(marks, 'bird'), birds: settle.mock.contexts.at(-1) as Flock, kept, part, stand }
}

/**
 * Fits a strip as `flown` does and paints it standing still, which takes the
 * picks at once.
 */
export function painted(murmur: Murmur = FLOCK, picks: readonly number[] = [0, 0, 0]): Flown {
  const strip = flown(murmur, picks)
  strip.part.paint(strip.stand.pen, geometry({ ...STRIP, still: true }), 1)
  return strip
}

/**
 * Reads how many strip pixels one pixel of the sky stands for, for a part
 * reaching `top` meters.
 */
export const scaleAt = (top: number): number => (stripY(20, 20, STRIP.h) - stripY(20, top, STRIP.h)) / 120
