import { idle, tween } from '@thermur/kit'

import { Marks }                 from '../canvas/marks.ts'
import { Stage }                 from '../canvas/stage.ts'
import { Tones }                 from '../canvas/tones.ts'
import { FIRE, MURMUR }          from '../front/drawing.ts'
import type { Identity, Scheme } from '../identity.ts'
import source                    from '../rendered/identity.json' with { type: 'json' }
import * as falcon               from './falcon.ts'

/**
 * How the drawing lays itself out and moves unless told otherwise.
 */
export const FALCON: falcon.Tuning = {
  birds  : [80, 160],
  fire   : FIRE,
  foot   : 78,
  inset  : 34,
  per    : 700,
  rate   : 10,
  reach  : 16,
  read   : 250,
  settle : MURMUR.settle,
  trace  : 12,
  zoom   : 1.35
}
/**
 * The icon beside each mode in the switch, as a path on a 16 by 16 grid, the
 * words it reads, and what the pointer does in it: a falcon stooping with its
 * wings swept back, and a pointer's arrow.
 */
export const MODE: Readonly<Record<falcon.Mode, { icon: string, title: string, words: string }>> = {
  predator: {
    icon  : 'M8 1.5 9.4 6.2 15 9.6 9.6 8.9 8.8 14.5 8 13 7.2 14.5 6.4 8.9 1 9.6 6.6 6.2Z',
    title : 'The pointer is the falcon',
    words : 'Predator'
  },
  select: {
    icon  : 'M3.5 1.5v12l3.2-3 2.3 5 2-.9-2.3-4.9h4.3Z',
    title : 'Press a bird to mark it',
    words : 'Select'
  }
}
/**
 * The fixed range the slider sets, in bird lengths.
 */
export const RANGE = { max: 6, min: 1, step: 0.25, value: 2 } as const

/**
 * The elements of the chip a drawing of the neighbors reads how far out the
 * marked bird's seventh flies into: the name, set in the seven's color, and
 * the value.
 */
export interface Reach {
  named : HTMLElement
  value : HTMLElement
}

/**
 * The drawing of the neighbors a drone senses: the front page's flock under
 * the pointer's falcon, its pointer's mode set through `mode` and its fixed
 * range through `range`, reading how far out the marked bird's seventh flies
 * into the chip `reach`. A new mark grows its lines to its seven out from it.
 * `destroy` releases every listener, observer, frame, and canvas it holds.
 */
export class Neighbors {
  #drawn           = -1
  readonly #reach  : Reach
  readonly #shared : falcon.Shared = {
    chosen : 0,
    grown  : 1,
    mode   : falcon.MODES[0],
    range  : RANGE.value
  }
  readonly #sky    : HTMLElement
  readonly #stage  : Stage
  #stop            = idle
  readonly #tones  : Tones

  constructor(
    host     : HTMLElement,
    reach    : Reach,
    identity : Identity      = source,
    tuning   : falcon.Tuning = FALCON
  ) {
    const tones            = new Tones(identity)
    const marks            = Marks.traced(tones, identity)
    const read             = (seventh: number, scheme: Scheme): void => this.#read(seventh, scheme)
    const part             = new falcon.Falcon(identity, marks, read, this.#shared, tones, tuning)
    this.#reach            = reach
    this.#sky              = host
    this.#tones            = tones
    this.#sky.dataset.mode = this.#shared.mode
    this.#stage            = new Stage(this.#sky, [part], { pinned: false })
    const canvas           = this.#sky.querySelector('canvas')!
    canvas.setAttribute('role', 'img')
    canvas.setAttribute('aria-label', 'The illustration flock in the logo’s birds, one bird marked with a '
                                    + 'line to each of its seven and a ring at the fixed range, above two '
                                    + 'strips counting the seven and the birds inside the ring over the last '
                                    + `${tuning.trace} seconds`)
  }

  /**
   * Sets what the pointer does in the drawing.
   */
  mode(mode: falcon.Mode): void {
    this.#shared.mode      = mode
    this.#sky.dataset.mode = mode
    this.#stage.repaint()
  }

  /**
   * Sets the fixed range, in bird lengths.
   */
  range(lengths: number): void {
    this.#shared.range = lengths
    this.#stage.repaint()
  }

  /**
   * Reads the seventh's distance into the chip, its name in the seven's color
   * in `scheme`, and grows the lines of a bird newly marked out to its seven.
   */
  #read(seventh: number, scheme: Scheme): void {
    const { chosen }              = this.#shared
    const moved                   = this.#drawn >= 0 && chosen !== this.#drawn
    this.#drawn                   = chosen
    this.#reach.named.style.color = this.#tones.flock(scheme, falcon.ALONG.seven, 1)
    this.#reach.value.textContent = `${seventh.toFixed(1)} lengths`
    if (!moved) return
    this.#stop()
    this.#stop = tween(share => {
      this.#shared.grown = share
      this.#stage.repaint()
    }, 2)
  }

  destroy(): void {
    this.#stop()
    this.#stage.destroy()
  }
}
