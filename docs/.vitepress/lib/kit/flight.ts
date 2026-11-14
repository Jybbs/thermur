import { markup } from '@thermur/kit'

import { lerp } from '../common.ts'

/**
 * The drawing's own units, the width and the height its viewBox spans.
 */
const WIDE = 600
const HIGH = 140

/**
 * How many birds the flock holds.
 */
const BIRDS = 36

/**
 * Where each bird sits in the loose flock and in the gathered one, as offsets
 * from the flock's center in the drawing's units, laid out once on a golden
 * angle so no two birds land on one point.
 */
const ANGLE     = Math.PI * (3 - Math.sqrt(5))
const PLACES    = Array.from({ length: BIRDS }, (_, bird) => {
  const turn   = bird * ANGLE
  const spread = Math.sqrt((bird + 0.5) / BIRDS)
  return { gathered: [Math.cos(turn) * spread * 22, Math.sin(turn) * spread * 14], loose: [Math.cos(turn) * spread * 150, Math.sin(turn) * spread * 46] }
})

/**
 * The catalog's sketch of the flock wheeling over the ground, its birds made
 * once and moved in place each frame, so a frame builds nothing.
 */
export class Flight {
  readonly node  : SVGElement
  readonly birds : SVGElement[]

  /**
   * The share the flock has gathered from loose to tight, from 0 to 1.
   */
  gathered = 0

  constructor() {
    this.birds = PLACES.map(() => markup.svg('circle', { fill: markup.TONE.ink, r: 2.4 }))
    this.node  = markup.svg('svg', { 'aria-hidden': 'true', 'class': 'kit-catalog-flight', 'viewBox': `0 0 ${WIDE} ${HIGH}` },
      markup.svg('line', { 'stroke': markup.TONE.rule, 'x1': 16, 'x2': WIDE - 16, 'y1': HIGH - 12, 'y2': HIGH - 12 }),
      ...this.birds)
    this.draw(0)
  }

  /**
   * Draws the flock `seconds` into its flight, its center sweeping along a
   * slow figure of eight and each bird wheeling about it, drawn between its
   * loose and gathered places by the share `gathered` holds.
   */
  draw(seconds: number): void {
    const centerX = WIDE / 2 + Math.sin(seconds * 0.4) * 160
    const centerY = HIGH / 2 - 6 + Math.sin(seconds * 0.8) * 18
    const wheel   = Math.sin(seconds * 0.6) * 0.5
    const [cos, sin] = [Math.cos(wheel), Math.sin(wheel)]
    this.birds.forEach((bird, place) => {
      const { gathered, loose } = PLACES[place]!
      const x = lerp(loose[0]!, gathered[0]!, this.gathered)
      const y = lerp(loose[1]!, gathered[1]!, this.gathered)
      bird.setAttribute('cx', (centerX + x * cos - y * sin).toFixed(1))
      bird.setAttribute('cy', (centerY + x * sin + y * cos).toFixed(1))
    })
  }
}
