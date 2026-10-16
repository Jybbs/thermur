import * as common                                from './common.ts'
import { Curve }                                  from './curve.ts'
import { blurs, type Fire, contours, fireground } from './fireground.ts'
import { colorOf, shadeOf }                       from './identity.ts'
import * as layout                                from './layout.ts'
import * as scene                                 from './scene.ts'
import { wakeOf }                                 from './wake.ts'

/**
 * What the main card is drawn from: the host and the version the edition names,
 * the traced birds of the mark as SVG path data, the dark scheme's small mark
 * and wordmark as data addresses satori reads, and the identity.
 */
interface MainCard extends layout.Signed {
  glyphs   : readonly string[]
  wordmark : string
}

/**
 * The palette color the wordmark's chevrons are drawn in, which the wake leaves
 * the wordmark in and the ring warms toward near the fire.
 */
const ACCENT = 'fire-lord-lift'
/**
 * The fire the main card shows from above, in pixels across and down the card,
 * stretched along the wind blowing toward its right edge.
 */
const FIRE: Fire = { across: 0.9, along: 1.25, puffs: 12, scale: 1.02, seed: 53, x: 905, y: 368 }
/**
 * The ground under the main card's terrain, a tone darker than the dark
 * scheme's ground, which the palette names no color for.
 */
const GROUND = '#121114'
/**
 * The points the flock's route runs through after the wordmark's tip, in pixels
 * across and down the card: climbing on at the chevrons' angle, then bending
 * over into a ring that enters the fire's left side, passes under it, and rises
 * up its right side to the smoke.
 */
const ROUTE: readonly common.Pair[] = [
  [492, 136], [512, 112], [538, 90], [572, 72], [616, 62], [670, 60], [728, 70], [782, 94], [826, 120],
  [830, 178], [790, 222], [722, 262], [676, 340], [684, 446], [762, 540], [905, 580], [1050, 546],
  [1142, 460], [1170, 380], [1160, 300]
]
/**
 * The veil of the card's ground behind the words, as stops across the card from
 * its left edge, fully dark at the edge and clear by its middle.
 */
const VEIL = [[0, 1], [0.36, 0.9], [0.5, 0]] as const
/**
 * The wordmark as its file draws it, in the file's own pixels: its width and
 * its height, the boxes its letters cover, meaning the x-height letters and the
 * tall first letter, and its trailing chevrons, each `chevron` long in rows
 * `rows` apart, the tip of the last at `tip`.
 */
const WORDMARK = {
  chevron : 14,
  height  : 216,
  letters : [{ x0: 0, x1: 636, y0: 124, y1: 199 }, { x0: 0, x1: 203, y0: 30, y1: 199 }],
  rows    : 20,
  tip     : [773.4, 15.6],
  width   : 790
} as const
/**
 * Where the main card sets its words, in pixels: the opacity of its edition,
 * the block's top, the wordmark `mark` wide, and the tagline beneath it after
 * `gap`, set at `size` across `width` at `opacity`, each line `leading` times
 * its size, wrapping onto `lines` lines.
 */
const WORDS = {
  edition : 0.7,
  gap     : 22,
  mark    : 420,
  tagline : { leading: 1.3, lines: 3, opacity: 0.82, size: 30, width: 480 },
  top     : 150
} as const
/**
 * The wordmark's height on the card, in pixels, at the width `WORDS` sets.
 */
const markHeight = Math.round(WORDS.mark * WORDMARK.height / WORDMARK.width)
/**
 * The card's pixels to one pixel of the wordmark's file.
 */
const markScale = WORDS.mark / WORDMARK.width

/**
 * Draws the main card's scene as an SVG's markup: the terrain and the fire seen
 * from above beneath a veil of the ground behind the words, and the flock over
 * the veil, leaving the wordmark's tip as a widening wake that runs on into a
 * ring around the fire.
 */
function backdrop({ glyphs, identity }: MainCard): string {
  const { height, inset, width } = layout.CANVAS
  const [across, down]           = WORDMARK.tip
  const cream                    = shadeOf('ink', 'dark', identity)
  const fire                     = fireground(FIRE, identity)
  const ground                   = shadeOf('ground', 'dark', identity)
  const stops                    = VEIL.map(([offset, opacity]) =>
    `<stop offset="${offset}" stop-color="${ground}" stop-opacity="${opacity}"/>`)
  const wake = wakeOf({
    boxes   : boxes(),
    chevron : WORDMARK.chevron * markScale,
    cream   : cream,
    fire    : fire,
    glyphs  : glyphs.length,
    lift    : colorOf(ACCENT, identity),
    route   : new Curve([[inset + across * markScale, WORDS.top + down * markScale], ...ROUTE]),
    rows    : WORDMARK.rows * markScale
  })

  return [
    `<defs>${scene.glyphDefs(glyphs)}${blurs}`,
    `<linearGradient id="veil" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="${width}" y2="0">`,
    `${stops.join('')}</linearGradient></defs>`,
    `<rect width="${width}" height="${height}" fill="${GROUND}"/>`,
    contours(cream),
    fire.svg,
    `<rect width="${width}" height="${height}" fill="url(#veil)"/>`,
    `<g>${wake}</g>`
  ].join('')
}

/**
 * Lists the boxes the words take on the card, which no bird enters: the
 * wordmark's letters and the tagline beneath them.
 */
function boxes(): common.Box[] {
  const { inset }   = layout.CANVAS
  const { tagline } = WORDS
  const top         = WORDS.top + markHeight + WORDS.gap
  const bottom      = top + tagline.lines * tagline.size * tagline.leading
  const letters     = WORDMARK.letters.map(box => ({
    x0 : inset + box.x0 * markScale,
    x1 : inset + box.x1 * markScale,
    y0 : WORDS.top + box.y0 * markScale,
    y1 : WORDS.top + box.y1 * markScale
  }))
  return [...letters, { x0: inset, x1: inset + tagline.width, y0: top, y1: bottom }]
}

/**
 * Lays out the main card, which the front page, the page met at a missing
 * address, and every page outside a section share: the wordmark and the tagline
 * over the scene, and the edition, on the dark scheme's ground.
 */
export function mainCard(card: MainCard): layout.Element {
  const { tagline } = WORDS
  return layout.card({
    backdrop : backdrop(card),
    column   : { gap: WORDS.gap, top: WORDS.top, width: Math.max(WORDS.mark, tagline.width) },
    edition  : { ...card, opacity: WORDS.edition },

    words: [
      layout.image(card.wordmark, WORDS.mark, markHeight),
      layout.element('div', {
        fontSize   : tagline.size,
        fontStyle  : 'italic',
        lineHeight : tagline.leading,
        opacity    : tagline.opacity,
        width      : tagline.width
      }, card.identity.tagline.text)
    ]
  })
}
