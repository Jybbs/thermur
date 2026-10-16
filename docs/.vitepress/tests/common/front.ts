import { vi } from 'vitest'

import { Marks }                      from '../../lib/canvas/marks.ts'
import type { Geometry }              from '../../lib/canvas/part.ts'
import { Tones }                      from '../../lib/canvas/tones.ts'
import { type Heat, repel }           from '../../lib/front/heat.ts'
import { Sky }                        from '../../lib/front/sky.ts'
import { type Browser, box, browser } from './browser.ts'

/**
 * An unfitted sky beside the marks and the tones it draws in and the stand-ins
 * it draws on.
 */
interface Unfitted {
  marks : Marks
  sky   : Sky
  stand : Browser
  tones : Tones
}

/**
 * The fire's strength a drawing's suite draws at.
 */
export const FIRE = 0.62
/**
 * A flock small enough and settled briefly enough for a case to fly many times
 * over.
 */
export const FLOCK = { count: 40, settle: 0.5 } as const

/**
 * Three traced birds, each a single point, which a drawing's suite draws its
 * flock from.
 */
const GLYPHS = ['M0 0Z', 'M1 1Z', 'M2 2Z']
/**
 * The places the front page's hero and reading column stand in the window, and
 * the page's length, in pixels.
 */
export const LAYOUT = {
  column : { bottom: 1500, left: 256, right: 944, top: 900 },
  hero   : { height: 640, top: 0 },
  page   : 2600
} as const

/**
 * Places the element `selector` finds at `rect` in the window.
 */
function place(selector: string, rect: Partial<DOMRect>): void {
  vi.spyOn(document.querySelector(selector)!, 'getBoundingClientRect').mockReturnValue(rect as DOMRect)
}

/**
 * Lays out a front page in the document, the hero and the reading column where
 * `LAYOUT` places them and the page as long as it sets, beside the box the
 * drawing fills, and returns the box.
 */
export function front(): HTMLElement {
  window.scrollTo(0, 0)
  const host = box('front-sky', '<div class="VPHero"></div><div class="VPHomeContent"></div>')
  place('.VPHero', LAYOUT.hero)
  place('.VPHomeContent', LAYOUT.column)
  vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockReturnValue(LAYOUT.page)
  return host
}

/**
 * Builds the page a part draws on: a window 1200 by 800 pixels over a page 3000
 * long, its hero ending 640 pixels down, in the dark scheme with no reading
 * column, with any of it replaced by `page`.
 */
export const geometry = (page: Partial<Geometry> = {}): Geometry => ({
  doc     : null,
  dy      : 0,
  h       : 800,
  hero    : { bottom: 640, h: 640, top: 0 },
  page    : 3000,
  pointer : null,
  scheme  : 'dark',
  still   : false,
  t       : 0,
  w       : 1200,
  ...page
})

/**
 * Reads the push away from `field`'s heat at a point at a strength of 1, the
 * heat there taken as `hot`, full unless given.
 */
export function pushed(field: Heat, x: number, y: number, hot = 1): [number, number] {
  const out: [number, number] = [0, 0]
  repel(field, x, y, hot, 1, out)
  return out
}

/**
 * Builds an unfitted sky at `FIRE` with `FLOCK` for its flock over the stand-in
 * browser, drawing from `glyphs`.
 */
export function skyOver(glyphs: readonly string[] = GLYPHS): Unfitted {
  const stand = browser()
  const tones = new Tones()
  const marks = new Marks(glyphs, tones)
  return { marks, sky: new Sky(marks, tones, FIRE, FLOCK), stand, tones }
}
