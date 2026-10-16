import type { Heat } from './heat.ts'

/**
 * The reader's pointer as a falcon over the sky, its heat falling off with the
 * distance from it as a Gaussian whose deviation is half of `radius`, and none
 * while it is off the sky.
 */
export class PointerHeat implements Heat {
  on = false
  x  = 0
  y  = 0
  readonly #radius : number

  constructor(radius: number) {
    this.#radius = radius
  }

  heat(x: number, y: number): number {
    if (!this.on) return 0
    return 0.9 * Math.exp(-((x - this.x) ** 2 + (y - this.y) ** 2) * 2 / this.#radius ** 2)
  }
}
