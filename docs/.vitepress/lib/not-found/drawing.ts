import { Marks }         from '../canvas/marks.ts'
import { Stage }         from '../canvas/stage.ts'
import { Tones }         from '../canvas/tones.ts'
import type { Identity } from '../identity.ts'
import source            from '../rendered/identity.json' with { type: 'json' }
import { Opening }       from './opening.ts'

/**
 * Draws the page met at a missing address into `host`: the murmuration
 * streaming across the page and parting around the 404, set in the page's flow.
 * The drawing fits again once the display face the numerals are set in has
 * loaded, since the numerals are measured in it.
 */
export function drawLost(host: HTMLElement, identity: Identity = source): Stage {
  const tones = new Tones(identity)
  const face  = identity.faces.display
  const stage = new Stage(host, [new Opening(Marks.traced(tones, identity), tones, face)], { pinned: false })
  void document.fonts.load(`400 1em '${face.family}'`).then(() => stage.refit())
  return stage
}
