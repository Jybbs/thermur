import { Marks }           from '../canvas/marks.ts'
import { Tones }           from '../canvas/tones.ts'
import source              from '../rendered/identity.json' with { type: 'json' }
import type { Player }     from './player.ts'
import { Scene }           from './scene.ts'
import { SHOT }            from './shot.ts'
import { READINGS, Strip } from './strip.ts'

/**
 * Builds each part the view draws, by the name of the box it fills: the scene
 * of the flock over the roost, which calls `shown` as each frame is painted,
 * and a strip for each reading.
 */
export function partsOf(player: Player, shown: () => void): Record<string, Scene | Strip> {
  const tones = new Tones(source)
  return {
    'scene': new Scene(Marks.traced(tones, source), tones, player, SHOT, shown),
    ...Object.fromEntries(READINGS.map((reading, index) => [`strip-${index}`, new Strip(player, reading, tones)]))
  }
}
