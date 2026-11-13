import fs   from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import manifest                                    from '../../../../kit/package.json' with { type: 'json' }
import { PARTS, PIECES, SOURCES, lineOf, pieceOf } from '../../../lib/kit/pieces.ts'
import { pagesDir }                                from '../../../lib/paths.ts'

/**
 * Reads the name in each row of the table The Kit carries, as `mise render`
 * last wrote it.
 */
function tabled(): string[] {
  const page = fs.readFileSync(path.join(pagesDir(import.meta.url), 'reference', 'kit.md'), 'utf8')
  return [...page.matchAll(/^\| (?<name>[^*|][^|]*) \| `/gmu)].map(row => row.groups!.name!)
}

describe('the kit’s pieces', () => {
  it('reads every declaration beside the kit’s code, in the order The Kit’s table lists them', () => {
    expect(PIECES.map(piece => piece.name)).toEqual(tabled())
    expect(PIECES.filter(piece => piece.machine !== null).map(piece => piece.machine))
      .toEqual(['radio-group', 'select', 'slider', 'toggle'])
  })

  it('files each piece under the part its folder names, the parts in path order', () => {
    expect(PARTS).toEqual(['chrome', 'controls', 'drawing', 'motion'])
    expect(pieceOf('gate')).toMatchObject({ exports: ['onMotion', 'pause', 'paused', 'still'], part: 'motion' })
  })

  it('reads each piece’s source as the files sharing its declaration’s name, each loading its text', async () => {
    expect(pieceOf('slider').sources).toEqual(['controls/slider.tsx'])
    expect(pieceOf('copy').sources).toEqual(['chrome/copy.ts'])
    expect(PIECES.every(piece => piece.sources.length > 0)).toBe(true)
    expect(await SOURCES['motion/timing.ts']!()).toContain('export function tween')
  })

  it('declares at least one example of each piece', () => {
    expect(PIECES.filter(piece => piece.examples.length === 0)).toEqual([])
  })

  it('writes the line a widget takes a piece up with, under the name the kit’s manifest declares', () => {
    expect(lineOf(pieceOf('gate'))).toBe(`import { onMotion, pause, paused, still } from '${manifest.name}'`)
  })

  it('refuses a key no piece carries', () => {
    expect(() => pieceOf('stray')).toThrow('the kit holds no piece stray')
  })
})
