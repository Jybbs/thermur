import type { Heat } from './heat.ts'

/**
 * How far the pointer's heat reaches, in pixels, twice the deviation of the
 * Gaussian it falls off along.
 */
const RADIUS = 80

/**
 * The reader's pointer as a falcon over the sky, its heat falling off with the
 * distance from it as a Gaussian, and none while it is off the sky.
 */
export class PointerHeat implements Heat {
  on = false
  x  = 0
  y  = 0

  heat(x: number, y: number): number {
    if (!this.on) return 0
    return 0.9 * Math.exp(-((x - this.x) ** 2 + (y - this.y) ** 2) * 2 / RADIUS ** 2)
  }
}
