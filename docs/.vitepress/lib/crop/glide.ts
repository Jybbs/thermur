/**
 * How fast a glide closes on its target, per second.
 */
const RATE = 3

/**
 * One number gliding to a target on a critically damped spring, integrated
 * exactly, so a new target taken mid-glide bends the path with no jump in where
 * the number stands or how fast it moves.
 */
export class Glide {
  target   : number
  value    : number
  velocity = 0

  constructor(value: number) {
    this.target = value
    this.value  = value
  }

  /**
   * Puts the number at its target at rest.
   */
  settle(): void {
    this.value    = this.target
    this.velocity = 0
  }

  /**
   * Moves the number on by `dt` seconds.
   */
  step(dt: number): void {
    const away    = this.value - this.target
    const push    = (this.velocity + RATE * away) * dt
    const fade    = Math.exp(-RATE * dt)
    this.velocity = (this.velocity - RATE * push) * fade
    this.value    = this.target + (away + push) * fade
  }
}
