import { describe, expect, it } from 'vitest'

import { body, crowds, fillOf, flock, spine, touches } from '../../../lib/sources/drawing.ts'
import type { Bird, Scene }                            from '../../../lib/sources/drawing.ts'

/**
 * A bird of no source, at the foot of the flock.
 */
const BIRD: Bird = { angle: 0, glyph: 0, length: 10, t: 0, x: 0, y: 0 }
/**
 * A list of five sources in a column 72 px wide down the section's left edge,
 * under a heading whose text runs 100 px across, the second source current.
 */
const SCENE: Scene = {
  rail    : { x0: 0, x1: 72, y0: 112, y1: 660 },
  row     : { x0: 0, x1: 640, y0: 0, y1: 72 },
  title   : { x0: 0, x1: 100, y0: 30, y1: 60 },
  width   : 640,
  anchors : [0, 1, 2, 3, 4].map(place => ({
    box     : { x0: 52, x1: 64, y0: 120 + place * 110, y1: 140 + place * 110 },
    current : place === 1
  }))
}
/**
 * `SCENE` with its list opening 8 px under the heading's row, where the row's
 * birds come nearest the first entry.
 */
const CLOSE: Scene = {
  ...SCENE,
  rail    : { ...SCENE.rail, y0: 80, y1: 628 },
  anchors : SCENE.anchors.map(({ box, current }) =>
    ({ box: { ...box, y0: box.y0 - 32, y1: box.y1 - 32 }, current }))
}
/**
 * The flock laid out over `SCENE`, from the mark's 28 traced birds.
 */
const laid = flock(SCENE, 28)
/**
 * The birds the numbers label, in the order of the sources.
 */
const numbered = laid.slice(0, SCENE.anchors.length)
/**
 * The birds no number labels.
 */
const loose = laid.slice(SCENE.anchors.length)

describe('flock', () => {
  it('lays a page out alike on every load at one width', () => {
    expect(flock(SCENE, 28)).toEqual(laid)
  })

  it('labels each number with a bird whose head sits just short of it', () => {
    expect(numbered.map(bird => bird.x)).toEqual(SCENE.anchors.map(anchor => anchor.box.x0 - 3))
    expect(numbered.map(bird => bird.current)).toEqual([false, true, false, false, false])
  })

  it('colors the flock from its foot at the last source to its head past the first', () => {
    const ts = numbered.map(bird => bird.t)
    expect(ts).toEqual(ts.toSorted((a, b) => b - a))
    expect(new Set(ts).size).toBe(ts.length)
  })

  it('flies a loose flock up the column and along the heading’s row', () => {
    expect(loose.length).toBeGreaterThan(40)
    expect(loose.some(bird => bird.y < SCENE.rail.y0)).toBe(true)
    expect(loose.some(bird => bird.x > SCENE.title.x1)).toBe(true)
  })

  it.for([['well under its heading', SCENE], ['just under its heading', CLOSE]] as const)(
    'keeps every bird in the column or the heading’s row, 4 px short of the entries, the list opening %s',
    ([, scene]) => {
      const astray = flock(scene, 28).slice(scene.anchors.length).filter(bird => {
        const { r, x, y } = body(bird)
        return Math.max(Math.min(x + r - scene.rail.x1, y + r - scene.rail.y0 + 4), x + r - scene.width) > 0
      })
      expect(astray).toEqual([])
    }
  )

  it('keeps every bird clear of the numbers, the heading’s text, and each other', () => {
    const boxes = [...SCENE.anchors.map(anchor => anchor.box), SCENE.title]
    expect(loose.filter(bird => boxes.some(box => touches(body(bird), box, 0)))).toEqual([])
    expect(loose.filter((bird, place) =>
      laid.toSpliced(place + SCENE.anchors.length, 1).some(other => crowds(bird, other, 0)))).toEqual([])
  })
})

describe('fillOf', () => {
  it('fills the bird of the current source in the lead hue', () => {
    expect(fillOf({ ...BIRD, current: true })).toBe('var(--th-lead)')
  })

  it.for([[0, '0.0'], [0.4, '50.0'], [0.9, '100.0']] as const)(
    'lays Fire Lord over the ink at %s along the flock by %s%%',
    ([t, share]) => {
      expect(fillOf({ ...BIRD, t })).toBe(`color-mix(in srgb, var(--th-series-1) ${share}%, var(--th-ink))`)
    }
  )
})

describe('spine', () => {
  const line = spine(SCENE.rail, 568, 50)

  it('runs from the foot of the column to the end of the heading’s row', () => {
    expect(line[0]!.along).toBe(0)
    expect(line.at(-1)).toMatchObject({ along: 1, x: 568, y: 50 })
    expect(line[0]!.y).toBe(SCENE.rail.y1 - 8)
  })

  it('keeps 40 points up a column too short for its pitch to fill', () => {
    expect(spine({ ...SCENE.rail, y1: SCENE.rail.y0 + 20 }, 568, 50)).toHaveLength(40 + 24 + 16)
  })

  it('samples a taller column more densely', () => {
    expect(spine({ ...SCENE.rail, y1: 4000 }, 568, 50).length).toBeGreaterThan(line.length)
  })
})

describe('touches', () => {
  it.for([
    [{ r: 5, x: 0, y: 0 }, false],
    [{ r: 5, x: 12, y: 12 }, true],
    [{ r: 5, x: 30, y: 20 }, true],
    [{ r: 5, x: 40, y: 20 }, false]
  ] as const)('reads the circle %o as touching the box: %s', ([circle, touching]) => {
    expect(touches(circle, { x0: 10, x1: 30, y0: 10, y1: 30 }, 2)).toBe(touching)
  })
})
