import { Stage }                 from '../canvas/stage.ts'
import { Tones }                 from '../canvas/tones.ts'
import { Lookout, type Readout } from './drawing.ts'
import type { Playing }          from './flight.ts'

/**
 * Draws a Lookout playing `playing` into `host` on a stage of its own, the
 * flight changed through `play`. `destroy` releases every listener, observer,
 * frame, and canvas it holds.
 */
export class LookoutStage {
  readonly #host  : HTMLElement
  readonly #part  : Lookout
  readonly #stage : Stage

  constructor(host: HTMLElement, playing: Playing, readout: Readout, tones = new Tones()) {
    this.#host  = host
    this.#part  = new Lookout(playing, readout, tones)
    this.#stage = new Stage(host, [this.#part], { pinned: false })
  }

  destroy(): void {
    this.#stage.destroy()
  }

  /**
   * Plays `playing` in place of the flight before, repainting at once.
   */
  play(playing: Playing): void {
    const { height, width } = this.#host.getBoundingClientRect()
    this.#part.play(playing, { h: height, w: width })
    this.#stage.repaint()
  }
}
