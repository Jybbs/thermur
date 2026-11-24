import { formatHex, rgb, wcagContrast } from 'culori'
import { describe, expect, it }         from 'vitest'

import type { Scheme } from '../../../lib/identity.ts'

import { Tones }         from '../../../lib/canvas/tones.ts'
import { NEIGHBORS }     from '../../../lib/common.ts'
import { seriesOf }      from '../../../lib/identity.ts'
import { ALONG, RANGED } from '../../../lib/rules/falcon.ts'
import { Strips }        from '../../../lib/rules/paint.ts'
import { FLOORS, laid }  from '../../common/contrast.ts'
import { geometry }      from '../../common/front.ts'

/**
 * One fill a paint lays down: its color and the opacity it is laid at.
 */
interface Fill {
  alpha : number
  style : string
}

/**
 * One word a paint writes, filled over the halo stroked under it.
 */
interface Word extends Fill {
  halo : string
  text : string
}

/**
 * What one paint of the strips lays down: every rectangle and every word.
 */
interface Painted {
  rects : Fill[]
  words : Word[]
}

/**
 * The two schemes every contrast holds under.
 */
const SCHEMES = ['dark', 'light'] as const

/**
 * Builds strips of four samples over the foot of a canvas 1200 pixels wide,
 * every sample but the newest counting five inside the range and seven by rank.
 */
function strips(tones: Tones): Strips {
  const built = new Strips('11px mono', 4, 4, tones)
  built.panel = { x0: 0, x1: 1200, y0: 720, y1: 800 }
  built.fill(5, NEIGHBORS)
  return built
}

/**
 * Paints `built` in `scheme` with the strips' colors `ranged` and `seven`, over
 * a context that writes down each rectangle and word it lays.
 */
function painted(built: Strips, scheme: Scheme, ranged: string, seven: string): Painted {
  const laidDown: Painted = { rects: [], words: [] }
  const pen               = {
    fillStyle   : '',
    globalAlpha : 1,
    strokeStyle : '',
    fillRect    : (): void => {
      laidDown.rects.push({ alpha: pen.globalAlpha, style: pen.fillStyle })
    },
    fillText    : (text: string): void => {
      laidDown.words.push({ alpha: pen.globalAlpha, halo: pen.strokeStyle, style: pen.fillStyle, text })
    },
    measureText : (): Partial<TextMetrics> => ({ width: 60 }),
    strokeText  : (): void => undefined
  }
  built.paint(pen as unknown as CanvasRenderingContext2D, geometry({ scheme }), ranged, seven)
  return laidDown
}

/**
 * Reads the color a fill paints over `under`, its style's own opacity and the
 * context's taken together.
 */
function over(under: string, { alpha, style }: Fill): string {
  const color = rgb(style)!
  return laid(under, formatHex(color), alpha * (color.alpha ?? 1))
}

describe('Strips', () => {
  it.each([
    ['dark', 3, 0.6 + 0.4 * 3 / NEIGHBORS],
    ['dark', 0, 0.6],
    ['light', 3, 0.9 + 0.1 * 3 / NEIGHBORS],
    ['light', 0, 0.9]
  ] as const)('labels the senders ranked in %s by their newest count, %i, shading it from the floor and the range from 0.8', (
    scheme,
    ranked,
    newest
  ) => {
    const built = strips(new Tones())
    built.sample(4, ranked)
    const { rects, words } = painted(built, scheme, 'ranged', 'seven')
    const shades           = (style: string): number[] => rects.filter(rect => rect.style === style).map(rect => rect.alpha)
    expect(words.map(word => word.text)).toEqual(expect.arrayContaining([`by rank, ${ranked}`, 'within range, 4']))
    expect(shades('seven')).toEqual([1, 1, 1, newest].map(alpha => expect.closeTo(alpha, 6)))
    expect(shades('ranged')).toEqual([5, 5, 5, 4].map(count => expect.closeTo(0.8 + 0.2 * count / NEIGHBORS, 6)))
  })

  it('writes each count’s word unchanged on a repeated paint at an unchanged count, and the new count’s once it changes', () => {
    const built = strips(new Tones())
    built.sample(4, 3)
    const texts = (): string[] => painted(built, 'dark', 'ranged', 'seven').words.map(word => word.text)
    const first = texts()
    expect(first).toEqual(['4 s ago', 'now', 'by rank, 3', 'within range, 4'])
    expect(texts()).toEqual(first)
    built.sample(9, 6)
    expect(texts()).toEqual(['4 s ago', 'now', 'by rank, 6', 'within range, 9'])
  })

  it('shades a range counting more than seven at full opacity', () => {
    const built = strips(new Tones())
    built.sample(12, 3)
    const { rects } = painted(built, 'light', 'ranged', 'seven')
    expect(rects.findLast(rect => rect.style === 'ranged')!.alpha).toBe(1)
  })
})

describe.for(SCHEMES)('the strips under the %s scheme', scheme => {
  const tones  = new Tones()
  const sky    = tones.sky(scheme)
  const ranged = seriesOf(RANGED, scheme)
  const seven  = tones.flock(scheme, ALONG.seven, 1)

  it.for(Array.from({ length: NEIGHBORS + 1 }, (_, count) => count))('hold a count of %i at a graphic’s contrast against the panel', count => {
    const built = strips(tones)
    built.sample(count, count)
    const [panel, ...rects] = painted(built, scheme, ranged, seven).rects
    const under             = over(sky.bands.at(-1)!, panel!)
    for (const style of [ranged, seven]) {
      const newest = rects.findLast(rect => rect.style === style)!
      expect(wcagContrast(over(under, newest), under)).toBeGreaterThanOrEqual(FLOORS.mark)
    }
  })

  it('write the time axis and the range’s count at a text’s contrast against their halo', () => {
    const built = strips(tones)
    built.sample(4, 3)
    const words = painted(built, scheme, ranged, seven).words.filter(word => !word.text.startsWith('by rank'))
    expect(words.map(word => word.text)).toEqual(['4 s ago', 'now', 'within range, 4'])
    for (const word of words) expect(wcagContrast(over(word.halo, word), word.halo)).toBeGreaterThanOrEqual(FLOORS.text)
  })
})
