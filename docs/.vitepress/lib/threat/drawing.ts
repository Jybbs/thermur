import { Marks }         from '../canvas/marks.ts'
import { Stage }         from '../canvas/stage.ts'
import { Tones }         from '../canvas/tones.ts'
import type { Identity } from '../identity.ts'
import source            from '../rendered/identity.json' with { type: 'json' }
import type { Front }    from './front.ts'
import type { Playback } from './playback.ts'
import { Plot }          from './plot.ts'
import type { Readout }  from './readings.ts'

/**
 * Draws the air along the ground as the threat's front passes into `host`, set
 * in the page's flow, fitting again once the faces its words are measured in
 * have loaded.
 *
 * Args:
 *   front    : The model of the threat the drawing runs.
 *   host     : The box the drawing fills.
 *   playback : The clock the drawing runs on, which the rail moves.
 *   readout  : The page's elements the drawing writes its reading into.
 *   identity : The identity whose colors and faces the drawing takes.
 */
export function drawThreat(
  front    : Front,
  host     : HTMLElement,
  playback : Playback,
  readout  : Readout,
  identity : Identity = source
): Stage {
  const tones = new Tones(identity)
  const marks = Marks.traced(tones, identity)
  const plot  = new Plot(front, identity, marks, playback, readout, tones)
  const stage = new Stage(host, [plot], { pinned: false })
  void document.fonts.load(`400 1em '${identity.faces.body.family}'`).then(() => stage.refit())
  return stage
}
