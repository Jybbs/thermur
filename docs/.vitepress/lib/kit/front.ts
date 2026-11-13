import { markup } from '@thermur/kit'

/**
 * The length of ground the catalog's front crosses, in meters, a made-up size.
 */
export const GROUND = 300

/**
 * The seconds the catalog's front burns for, end to end.
 */
export const SPAN = 120

/**
 * The seconds between two snapshots of the catalog's front.
 */
export const SNAPSHOT = 15

/**
 * The fuels the front burns through, each beside the made-up rate it spreads
 * at in still air, in meters a second.
 */
export const FUELS = [
  { rate: 1.2, words: 'dry grass' },
  { rate: 0.6, words: 'brush' },
  { rate: 0.3, words: 'timber litter' }
] as const

/**
 * The winds pushing the front, each beside the made-up share of the still-air
 * rate it spreads at.
 */
export const WINDS = [
  { push: 0.5, value: 'calm', words: 'Calm' },
  { push: 1, value: 'breeze', words: 'Breeze' },
  { push: 2, value: 'gale', words: 'Gale' }
] as const

/**
 * Reads how far the front has spread through the fuel at `fuel` under the
 * wind at `wind` after `seconds`, in meters, stopping at the ground's end.
 */
export const reachOf = (fuel: number, wind: number, seconds: number): number =>
  Math.min(GROUND, FUELS[fuel]!.rate * WINDS[wind]!.push * seconds)

/**
 * The drawing's height and its ground line, in pixels, its width taking the
 * column it stands in, and the margin either end of the ground keeps.
 */
const HIGH   = 112
const FLOOR  = 84
const MARGIN = 16

/**
 * The narrowest gap, in pixels, between two ruler labels a drawing writes
 * both of, below which it writes every other one.
 */
const ROOM = 48

/**
 * The birds flying ahead of the front, each as its distance ahead of it in
 * meters and its height above the ground in pixels.
 */
const BIRDS = [[24, 34], [30, 44], [36, 30], [40, 52], [46, 38], [52, 46], [58, 34]] as const

/**
 * The meters the ruler marks, every fifty from ignition to the ground's end.
 */
const MARKS = Array.from({ length: GROUND / 50 + 1 }, (_, place) => place * 50)

/**
 * The drawing count, so each drawing's gradient takes an id of its own.
 */
let drawn = 0

/**
 * The catalog's sketch of a fire front crossing the ground, the burned ground
 * behind it in the heat ramp, a ruler beneath it in meters, and the flock
 * flying a little ahead of it, all made once and moved by `draw`. `fit` lays
 * it out at the width it stands at, so its labels keep their size in pixels
 * on a phone and in a wide column alike.
 */
export class Front {
  readonly node   : SVGElement
  readonly burn   : SVGElement
  readonly edge   : SVGElement
  readonly flock  : SVGElement
  readonly birds  : SVGElement[]
  readonly ground : SVGElement
  readonly ticks  : SVGElement[]
  readonly labels : SVGElement[]
  wide            = 600
  reach           = 0

  constructor() {
    drawn += 1
    const id    = `kit-front-${drawn}`
    const ramp  = markup.svg('linearGradient', { id }, ...[6, 5, 4, 3, 2, 1].map((step, place) =>
      markup.svg('stop', { offset: place / 5, style: `stop-color: var(--th-heat-${step})` })))
    this.burn   = markup.svg('rect', { fill: `url(#${id})`, height: 10, rx: 2, x: MARGIN, y: FLOOR - 10 })
    this.edge   = markup.svg('line', { 'stroke': 'var(--th-heat-2)', 'stroke-linecap': 'round', 'stroke-width': 3, 'y1': FLOOR - 22, 'y2': FLOOR })
    this.birds  = BIRDS.map(([, high]) => markup.svg('circle', { cy: FLOOR - high, fill: markup.TONE.ink, r: 2.2 }))
    this.flock  = markup.svg('g', {}, ...this.birds)
    this.ground = markup.svg('line', { 'stroke': markup.TONE.rule, 'x1': MARGIN, 'y1': FLOOR, 'y2': FLOOR })
    this.ticks  = MARKS.map(() => markup.svg('line', { 'stroke': markup.TONE.faint, 'y1': FLOOR + 4, 'y2': FLOOR + 8 }))
    this.labels = MARKS.map(meters => markup.text(markup.FACE.mono, 10, `${meters} m`, 0, FLOOR + 22, { 'fill': markup.TONE.muted, 'text-anchor': 'middle' }))
    this.node   = markup.svg('svg', { 'aria-hidden': 'true', 'class': 'kit-catalog-front', 'height': HIGH },
      markup.svg('defs', {}, ramp),
      this.ground, ...this.ticks, ...this.labels, this.burn, this.edge, this.flock,
      markup.text(markup.FACE.mono, 10, 'MADE-UP SPREAD RATES', MARGIN, 14, { 'fill': markup.TONE.faint, 'letter-spacing': '0.1em' }))
    this.fit(this.wide)
  }

  /**
   * The pixels a meter of ground spans at the width the drawing stands at.
   */
  get scale(): number {
    return (this.wide - 2 * MARGIN) / GROUND
  }

  /**
   * Places `meters` along the ground, in pixels from the drawing's left edge.
   */
  xOf(meters: number): number {
    return MARGIN + meters * this.scale
  }

  /**
   * Lays the drawing out `width` pixels wide, writing every ruler label where
   * two fit `ROOM` apart and every other one where they do not, and draws the
   * front again where it stood.
   */
  fit(width: number): void {
    this.wide   = width
    const every = 50 * this.scale < ROOM ? 2 : 1
    markup.set(this.node, { viewBox: `0 0 ${width} ${HIGH}` })
    markup.set(this.ground, { x2: width - MARGIN })
    MARKS.forEach((meters, place) => {
      markup.set(this.ticks[place]!, { x1: this.xOf(meters), x2: this.xOf(meters) })
      markup.set(this.labels[place]!, { display: place % every === 0 ? 'inline' : 'none', x: this.xOf(meters) })
    })
    BIRDS.forEach(([ahead], place) => markup.set(this.birds[place]!, { cx: ahead * this.scale }))
    this.draw(this.reach)
  }

  /**
   * Moves the front to `reach` meters along the ground, the burned ground
   * reaching it and the flock flying ahead of it until the flock meets the
   * ground's end.
   */
  draw(reach: number): void {
    this.reach  = reach
    const ahead = Math.min(reach, GROUND - BIRDS.at(-1)![0])
    markup.set(this.burn, { width: reach * this.scale })
    markup.set(this.edge, { x1: this.xOf(reach), x2: this.xOf(reach) })
    markup.set(this.flock, { transform: `translate(${this.xOf(ahead)} 0)` })
  }
}
