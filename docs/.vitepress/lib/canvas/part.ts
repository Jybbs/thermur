import type { Pair }   from '../common.ts'
import type { Scheme } from '../identity.ts'

/**
 * The reading column of the page, in pixels across the window and down the
 * page.
 */
interface Column {
  bottom : number
  left   : number
  right  : number
  top    : number
}

/**
 * The page as a part draws it this frame: the canvas's width and height, the
 * page's scroll and length, the hero's box and the reading column down the
 * page, the scheme, the clock, the reader's pointer on the page for a pinned
 * stage and on the canvas for one in the page's flow, and whether the drawing
 * stands still. The stage updates one record in place, so a part reads it only
 * for the frame it is handed in.
 */
export interface Geometry {
  doc     : Column | null
  dy      : number
  h       : number
  hero    : { bottom: number, h: number, top: number }
  page    : number
  pointer : Pair | null
  scheme  : Scheme
  still   : boolean
  t       : number
  w       : number
}

/**
 * One drawing on the stage, fitted to the page whenever the page changes size,
 * stepped on each frame while motion runs, painted on each frame and whenever
 * the page scrolls or the scheme changes while it stands still, and told of
 * each click on the stage's canvas.
 */
export interface Part {
  paint  : (ctx: CanvasRenderingContext2D, g: Geometry, dpr: number) => void
  fit?   : (g: Geometry) => void
  press? : (g: Geometry) => void
  step?  : (dt: number, g: Geometry) => void
}

/**
 * The width below which a drawing lays itself out for a narrow window, in
 * pixels.
 */
export const NARROW = 640
