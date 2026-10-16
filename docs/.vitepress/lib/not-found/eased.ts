import type { Geometry } from '../canvas/part.ts'
import type { Pair }     from '../common.ts'

/**
 * The reader's pointer as the drawing reads it, eased toward where the pointer
 * is, so a sweep glides rather than jumps, and fading in as the pointer arrives
 * and out after it leaves.
 */
export class EasedPointer {
  on = 0
  x  = 0
  y  = 0

  /**
   * Moves the eased pointer straight to a pointer at `x` and `y`.
   */
  #jump([x, y]: Pair): void {
    this.x = x
    this.y = y
  }

  /**
   * Places the pointer for one frame, eased while motion runs and at the
   * pointer itself, fully on or off, while the drawing stands still.
   */
  frame({ pointer, still }: Pick<Geometry, 'pointer' | 'still'>): this {
    if (!still) return this
    this.on = pointer === null ? 0 : 1
    if (pointer !== null) this.#jump(pointer)
    return this
  }

  /**
   * Eases the pointer `dt` seconds on toward `pointer`, starting at it where
   * the pointer has only just arrived, or fades it out while there is none.
   */
  step(dt: number, pointer: Pair | null): void {
    if (pointer === null) {
      this.on *= Math.exp(-dt * 3)
      return
    }
    if (this.on < 0.02) this.#jump(pointer)
    const ease = 1 - Math.exp(-dt * 8)
    this.x    += (pointer[0] - this.x) * ease
    this.y    += (pointer[1] - this.y) * ease
    this.on   += (1 - this.on) * (1 - Math.exp(-dt * 5))
  }
}
