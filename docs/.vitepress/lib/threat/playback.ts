import type { Front } from './front.ts'

/**
 * The seconds of the threat's clock one second of the reader's runs through.
 */
const PACE = 10
/**
 * How long the run before fades as the next front enters, in the reader's
 * seconds.
 */
const FADE = 1.2
/**
 * How fast the drawing glides to the time a reader set, and how fast the grip
 * swells under a press, each per second.
 */
const GLIDE = 9
/**
 * How long a reader's last move on the rail holds the run, in seconds.
 */
const HOLD = 4
/**
 * How long a finished run rests before the next front enters, in seconds.
 */
const REST = 1.6

/**
 * The clock a drawing of the front runs on: the seconds it draws, which run on
 * their own at a steady pace, rest at the end of the run, and start over as the
 * run before fades, so the front never runs backward. A reader's move on the
 * rail sets the time the drawing glides to and holds the run until a few
 * seconds after the move, where a drawing standing still takes the time at
 * once.
 */
export class Playback {
  fading           = 0
  grip             = 0
  seconds          = 0
  readonly #front  : Front
  #held            = 0
  #moved           = false
  #pressed         = false
  #resting         = 0
  #target          = 0

  constructor(front: Front) {
    this.#front = front
  }

  /**
   * Reads the time the drawing is heading for, the one a reader set or else the
   * one it draws.
   */
  get target(): number {
    return this.#target
  }

  /**
   * Holds the run for as long as a reader presses the rail.
   */
  press(): void {
    this.#pressed = true
  }

  /**
   * Lets the hold run out from the moment a reader lets the rail go.
   */
  release(): void {
    this.#pressed = false
    this.#held    = HOLD
  }

  /**
   * Sets the time the drawing glides to, held between the start and the end of
   * the run.
   */
  seek(seconds: number): void {
    this.#target  = Math.min(this.#front.span, Math.max(0, seconds))
    this.#held    = HOLD
    this.#moved   = true
    this.#resting = 0
    this.fading   = 0
  }

  /**
   * Takes the time a reader set at once, where the drawing stands still, or the
   * middle of the run where it has not moved, since an empty plot shows
   * nothing.
   */
  settle(): void {
    if (!this.#moved) this.#target = this.#front.span / 2
    this.#moved  = true
    this.seconds = this.#target
    this.grip    = this.#pressed ? 1 : 0
  }

  /**
   * Runs the clock on by `dt` of the reader's seconds.
   */
  step(dt: number): void {
    const glide  = 1 - Math.exp(-GLIDE * dt)
    this.#moved  = true
    this.fading  = Math.max(0, this.fading - dt / FADE)
    this.grip   += ((this.#pressed ? 1 : 0) - this.grip) * glide
    if (this.#pressed || this.#held > 0) {
      this.#held    = this.#pressed ? this.#held : Math.max(0, this.#held - dt)
      this.seconds += (this.#target - this.seconds) * glide
      return
    }
    if (this.seconds < this.#front.span) {
      this.seconds = Math.min(this.#front.span, this.seconds + PACE * dt)
      this.#target = this.seconds
      return
    }
    this.#resting += dt
    if (this.#resting < REST) return
    this.#resting = 0
    this.#target  = 0
    this.fading   = 1
    this.seconds  = 0
  }
}
