import { Marks }                 from '../canvas/marks.ts'
import { Stage }                 from '../canvas/stage.ts'
import { Tones }                 from '../canvas/tones.ts'
import { FIRE, MURMUR }          from '../front/drawing.ts'
import type { Identity, Scheme } from '../identity.ts'
import source                    from '../rendered/identity.json' with { type: 'json' }
import * as controls             from './controls.ts'
import * as falcon               from './falcon.ts'
import { tween }                 from './motion.ts'

/**
 * How the drawing lays itself out and moves unless told otherwise.
 */
const FALCON: falcon.Tuning = {
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
const MODE: Readonly<Record<falcon.Mode, { icon: string, title: string, words: string }>> = {
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
const RANGE = { max: 6, min: 1, step: 0.25, value: 2 } as const
/**
 * The SVG namespace the switch's icons are made in.
 */
const SVG = 'http://www.w3.org/2000/svg'

/**
 * Does nothing, standing in for a growth of the lines before the first one
 * starts.
 */
const idle = (): void => {}

/**
 * Makes the switch between predator mode and select mode, a radio group whose
 * thumb slides under the mode that is on, each mode beside its icon and each
 * button titled with what the pointer does in it.
 */
function modeSwitch(initial: falcon.Mode, picked: (mode: falcon.Mode) => void): HTMLElement {
  const thumb = document.createElement('span')
  const words = falcon.MODES.map(mode => [mode, MODE[mode].words] as const)
  const group = controls.radios(initial, 'Pointer', words, mode => {
    group.dataset.mode = mode
    picked(mode)
  })
  for (const button of group.querySelectorAll<HTMLElement>('[role="radio"]')) {
    const { icon, title } = MODE[button.dataset.value as falcon.Mode]
    const glyph           = document.createElementNS(SVG, 'svg')
    const path            = document.createElementNS(SVG, 'path')
    button.title          = title
    glyph.setAttribute('viewBox', '0 0 16 16')
    glyph.setAttribute('aria-hidden', 'true')
    path.setAttribute('d', icon)
    glyph.append(path)
    button.prepend(glyph)
  }
  thumb.className    = 'rules-mode-thumb'
  group.dataset.mode = initial
  group.classList.add('rules-mode')
  thumb.setAttribute('aria-hidden', 'true')
  group.prepend(thumb)
  return group
}

/**
 * The drawing of the neighbors a drone senses: a header naming the strips
 * beside the switch between the pointer's modes, the slider setting the fixed
 * range, and the chip reading how far out the marked bird's seventh flies,
 * above the front page's flock under the pointer's falcon. A new mark grows
 * its lines to its seven out from it. `destroy` releases every listener,
 * observer, frame, and canvas it holds.
 */
export class Neighbors {
  #drawn           = -1
  readonly #reach  = controls.chip('Seventh by rank')
  readonly #shared : falcon.Shared = {
    chosen : 0,
    grown  : 1,
    mode   : falcon.MODES[0],
    range  : RANGE.value
  }
  readonly #sky    = document.createElement('div')
  readonly #stage  : Stage
  #stop            = idle
  readonly #tones  : Tones

  constructor(host: HTMLElement, identity: Identity = source, tuning: falcon.Tuning = FALCON) {
    const tones            = new Tones(identity)
    const marks            = Marks.traced(tones, identity)
    const read             = (seventh: number, scheme: Scheme): void => this.#read(seventh, scheme)
    const part             = new falcon.Falcon(identity, marks, read, this.#shared, tones, tuning)
    this.#tones            = tones
    this.#sky.className    = 'rules-stage'
    this.#sky.dataset.mode = this.#shared.mode
    host.replaceChildren(this.#header(tuning.trace), this.#sky)
    this.#stage            = new Stage(this.#sky, [part], { pinned: false })
    const canvas           = this.#sky.querySelector('canvas')!
    canvas.setAttribute('role', 'img')
    canvas.setAttribute('aria-label', 'The illustration flock in the logo’s birds, one bird marked with a '
                                    + 'line to each of its seven and a ring at the fixed range, above two '
                                    + 'strips counting the seven and the birds inside the ring over the last '
                                    + `${tuning.trace} seconds`)
  }

  /**
   * Builds the header above the drawing: its title naming the strips over the
   * last `seconds`, the switch between the pointer's modes, the slider setting
   * the fixed range, and the chip reading the seventh's distance.
   */
  #header(seconds: number): HTMLElement {
    const bar         = document.createElement('div')
    const title       = document.createElement('span')
    const mode        = modeSwitch(this.#shared.mode, picked => {
      this.#shared.mode      = picked
      this.#sky.dataset.mode = picked
      this.#stage.repaint()
    })
    const range       = controls.slider('Range, in bird lengths', RANGE, 'Range', value => {
      this.#shared.range = value
      this.#stage.repaint()
    })
    bar.className     = 'rules-bar'
    title.className   = 'rules-title'
    title.textContent = `Birds around the marked bird, last ${seconds} s`
    bar.append(title, mode, range, this.#reach.element)
    return bar
  }

  /**
   * Reads the seventh's distance into the chip, its name in the seven's color
   * in `scheme`, and grows the lines of a bird newly marked out to its seven.
   */
  #read(seventh: number, scheme: Scheme): void {
    const { chosen }              = this.#shared
    const moved                   = this.#drawn >= 0 && chosen !== this.#drawn
    this.#drawn                   = chosen
    this.#reach.key.style.color   = this.#tones.flock(scheme, falcon.ALONG.seven, 1)
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
