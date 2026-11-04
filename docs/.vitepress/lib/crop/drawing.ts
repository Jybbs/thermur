import { Marks }         from '../canvas/marks.ts'
import { Stage }         from '../canvas/stage.ts'
import { Tones }         from '../canvas/tones.ts'
import { FIRE, MURMUR }  from '../front/drawing.ts'
import type { Murmur }   from '../front/sky.ts'
import type { Identity } from '../identity.ts'
import source            from '../rendered/identity.json' with { type: 'json' }
import type { Kept }     from './kept.ts'
import { FlightStrip }   from './strip.ts'

/**
 * Draws the part of an output the menus keep into `host`, set in the page's
 * flow.
 *
 * Args:
 *   host     : The box the drawing fills.
 *   kept     : What the menus keep.
 *   identity : The identity whose colors and marks the drawing takes.
 *   murmur   : The flock flying inside the part.
 */
export function drawKept(
  host     : HTMLElement,
  kept     : Kept,
  identity : Identity = source,
  murmur   : Murmur   = MURMUR
): Stage {
  const tones = new Tones(identity)
  const marks = Marks.traced(tones, identity)
  return new Stage(host, [new FlightStrip(kept, marks, tones, FIRE, murmur)], { pinned: false })
}
