import { type Identity, fontStack } from '../identity.ts'

/**
 * The numerals the drawing sets at its middle.
 */
const TEXT = '404'

/**
 * The 404 set in the display face at the middle of the drawing, at a size a
 * third of the drawing's width up to 230 pixels, measured once each time the
 * drawing fits its box, so a frame sets it without measuring.
 */
export class Numerals {
  cx                = 0
  cy                = 0
  height            = 0
  width             = 0
  #drop             = 0
  readonly #family  : string
  #font             = ''
  readonly #measure : CanvasRenderingContext2D

  constructor(face: Identity['faces']['display']) {
    this.#family  = fontStack(face)
    this.#measure = document.createElement('canvas').getContext('2d')!
  }

  /**
   * Measures the numerals for a drawing `w` by `h` pixels: their ink's width
   * and height, and how far below their middle their baseline sits.
   */
  fit(w: number, h: number): void {
    this.#font         = `400 ${Math.round(Math.min(w * 0.34, 230))}px ${this.#family}`
    this.#measure.font = this.#font
    const metrics      = this.#measure.measureText(TEXT)
    const ascent       = metrics.actualBoundingBoxAscent
    const descent      = metrics.actualBoundingBoxDescent

    this.cx     = w / 2
    this.cy     = h / 2
    this.width  = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight
    this.height = ascent + descent
    this.#drop  = (ascent - descent) / 2
  }

  /**
   * Sets the numerals at the middle of the drawing in `color`.
   */
  paint(ctx: CanvasRenderingContext2D, color: string): void {
    ctx.font         = this.#font
    ctx.textAlign    = 'center'
    ctx.textBaseline = 'alphabetic'
    ctx.fillStyle    = color
    ctx.fillText(TEXT, this.cx, this.cy + this.#drop)
  }
}
