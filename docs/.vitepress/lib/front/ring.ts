import type { Heat } from './heat.ts'

/**
 * A fire the eight lines part around, its heat strongest at its center and
 * falling to nothing at `radius` pixels from it, its strength the fire's.
 */
export class Ring implements Heat {
  readonly fire   : number
  readonly radius : number
  readonly x      : number
  readonly y      : number

  constructor({ fire, radius, x, y }: { fire: number, radius: number, x: number, y: number }) {
    this.fire   = fire
    this.radius = radius
    this.x      = x
    this.y      = y
  }

  heat(px: number, py: number): number {
    const d = Math.hypot(px - this.x, py - this.y)
    return this.fire <= 0 || d >= this.radius ? 0 : 1.8 * this.fire * (1 - d / this.radius) ** 2
  }
}
