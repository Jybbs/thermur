import { Marks }            from '../canvas/marks.ts'
import { Stage }            from '../canvas/stage.ts'
import { Tones }            from '../canvas/tones.ts'
import type { Identity }    from '../identity.ts'
import source               from '../rendered/identity.json' with { type: 'json' }
import { Lines }            from './lines.ts'
import { type Murmur, Sky } from './sky.ts'

/**
 * The fire's strength the front page draws, on a made-up scale from out at 0 to
 * a blaze at 1.
 */
const FIRE = 0.62
/**
 * The flock the front page draws.
 */
const MURMUR: Murmur = { count: 340, settle: 8 }

/**
 * Draws the front page's background into `host`: the banded sky over the hero
 * with `murmur` its flock, and the eight lines running down the page beneath it
 * to the roost at its foot, the traced birds read from the dark scheme's mark.
 */
export function drawFront(
  host     : HTMLElement,
  identity : Identity = source,
  murmur   : Murmur   = MURMUR
): Stage {
  const tones = new Tones(identity)
  const marks = Marks.traced(tones, identity)
  const sky   = new Sky(marks, tones, FIRE, murmur)
  return new Stage(host, [sky, new Lines(sky, marks, tones, FIRE)], { pinned: true })
}
