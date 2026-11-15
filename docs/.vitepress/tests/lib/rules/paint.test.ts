import { describe, expect, it, vi } from 'vitest'

import type { Scheme } from '../../../lib/identity.ts'

import { Tones }     from '../../../lib/canvas/tones.ts'
import { NEIGHBORS } from '../../../lib/common.ts'
import { Strips }    from '../../../lib/rules/paint.ts'
import { geometry }  from '../../common/front.ts'

/**
 * Paints strips of four samples whose newest counts `within` inside the range
 * and `ranked` by rank in `scheme`, over a context that writes down each word
 * it fills and the opacity each rectangle in the color `seven` is filled at.
 */
function painted(ranked: number, scheme: Scheme, within: number): { alphas: number[], words: string[] } {
  const alphas: number[] = []
  const words: string[]  = []
  const pen              = {
    fillStyle   : '',
    globalAlpha : 1,
    fillRect    : (): void => {
      if (pen.fillStyle === 'seven') alphas.push(pen.globalAlpha)
    },
    fillText    : (text: string): number => words.push(text),
    measureText : (): Partial<TextMetrics> => ({ width: 60 }),
    strokeText  : vi.fn<() => void>()
  }
  const ctx    = pen as unknown as CanvasRenderingContext2D
  const strips = new Strips('11px mono', 4, 4, new Tones())
  strips.panel = { x0: 0, x1: 1200, y0: 720, y1: 800 }
  strips.fill(5, NEIGHBORS)
  strips.sample(within, ranked)
  strips.paint(ctx, geometry({ scheme }), 'ranged', 'seven')
  return { alphas, words }
}

describe('Strips', () => {
  it.each([
    ['dark', 3, 0.6 + 0.25 * 3 / NEIGHBORS],
    ['dark', 0, 0.6],
    ['light', 3, 0.15 + 0.7 * 3 / NEIGHBORS],
    ['light', 0, 0.15]
  ] as const)('labels the senders ranked in %s by their newest count, %i, shading it from the floor', (
    scheme,
    ranked,
    newest
  ) => {
    const { alphas, words } = painted(ranked, scheme, 4)
    expect(words).toContain(`by rank, ${ranked}`)
    expect(words).toContain('within range, 4')
    expect(alphas.slice(-4)).toEqual([0.85, 0.85, 0.85, newest].map(alpha => expect.closeTo(alpha, 6)))
  })
})
