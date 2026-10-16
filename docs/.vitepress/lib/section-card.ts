import * as common                                    from './common.ts'
import { type Identity, heatRamp, seriesOf, shadeOf } from './identity.ts'
import * as layout                                    from './layout.ts'
import * as scene                                     from './scene.ts'
import { type Placed, numeralOf }                     from './sections.ts'

/**
 * What a section card is drawn from: the page's title and the section it sits
 * in, the host and the version the edition names, the traced birds of the mark
 * as SVG path data, the dark scheme's small mark as a data address satori
 * reads, and the identity.
 */
export interface SectionCard extends layout.Signed {
  glyphs   : readonly string[]
  section  : Placed
  title    : string
}

/**
 * The tones a night scene draws in that the palette names no color for: `gray`,
 * the neutral a section's hue is softened toward, and three tones darker than
 * the dark scheme's ground, meaning `ridge`, the ridge in front, `silhouette`,
 * the flock against the sky, and `sky`, the night above the sky's bands.
 */
const NIGHT = { gray: '#77767c', ridge: '#0b0a0c', silhouette: '#0e0d10', sky: '#131216' } as const
/**
 * A section card's sky, in pixels down and across the card: the horizon the
 * bands run down to, the fire's place along the ridge, the ridge's waves eight
 * pixels below the horizon, and the share of the way down to the horizon the
 * four bands of the section's hue start at. The four thinner bands above them
 * stack from the bottom up at `heights`, each filled `shares` of the way from
 * the night to the tone of the band beneath the stack.
 */
const SKY = {
  feather : { heights: [30, 23, 17, 12], shares: [0.62, 0.36, 0.17, 0.06] },
  fire    : 800,
  horizon : 470,
  ridge   : { amplitude: 18, periods: [140, 61, 23], phases: [1.1, 2.2, 3.3] },
  start   : 0.58
} as const
/**
 * The words a section card sets, in pixels: the opacity its edition is set at,
 * the block's top and width, the size its label is set at, and the size of its
 * title, the first whose `chars` the title's length does not pass.
 */
const WORDS = {
  edition : 0.75,
  label   : 20,
  titles  : [{ chars: 14, size: 88 }, { chars: 26, size: 74 }, { chars: Infinity, size: 60 }],
  top     : 70,
  width   : 520
} as const
/**
 * The filter a glow is drawn through, a Gaussian blur with a standard deviation
 * of 30 pixels over a region reaching past the shape it blurs.
 */
const glow = scene.blur('glow', 30, 300)

/**
 * Finds where across the card the plume stands `y` pixels down, leaning
 * downwind as it rises from the fire.
 */
const plumeAt = (y: number): number => SKY.fire + 0.5 * (SKY.horizon - y)

/**
 * Reports how deep in the plume's heat a bird at a point sits: 2 inside the
 * plume, where no bird flies, and outside it a share falling from one half at
 * the plume's edge to nothing past it.
 */
function plumeHeat(x: number, y: number): number {
  const reach = Math.abs(x - plumeAt(y)) / (1.1 * (24 + 0.2 * (SKY.horizon - y)))
  return scene.depth(reach, 1, 1.5)
}

/**
 * Measures how far the plume raises a band's edge at a point, up to 18 pixels
 * at the plume's center and less toward its sides and its foot.
 */
function plumeLift(x: number, y: number): number {
  const rise   = SKY.horizon - y
  const spread = 34 + 0.25 * rise
  const fade   = 0.25 + 0.75 * Math.sin(Math.PI * Math.min(1, rise / SKY.horizon + 0.1))
  return 18 * Math.exp(-((x - plumeAt(y)) ** 2) / (2 * spread ** 2)) * fade
}

/**
 * Draws the sky of the section at `section.place` as an SVG's markup: its bands
 * lifting over the plume, the fire's glow and its line on the ridge, and the
 * flock crossing low as a silhouette, parted where the plume rises.
 */
function sky({ glyphs, identity, section }: SectionCard): string {
  const { height, width } = layout.CANVAS
  const { fire, horizon } = SKY
  const heat              = heatRamp(identity)
  const ridge             = common.ridgeline({ ...SKY.ridge, base: horizon + 8 })
  const rim               = (x: number): number => scene.tenths(ridge(x))
  const crest             = scene.tenths(ridge(fire) - 6)
  const front             = `M${fire - 46} ${rim(fire - 46)}Q${fire} ${crest} ${fire + 46} ${rim(fire + 46)}`
  const flock             = scene.murmuration({
    color  : () => NIGHT.silhouette,
    count  : 900,
    gap    : plumeHeat,
    glyphs : glyphs.length,
    seed   : 41 + section.place,
    size   : (_, draw) => 4.5 + 4 * draw ** 1.6,
    width  : along => 22 + 30 * Math.sin(along * Math.PI),

    spine: along => [
      520 + along * 700,
      360 - 50 * Math.sin(along * Math.PI * 1.1) + 22 * Math.sin(along * 6.5)
    ]
  })

  return [
    `<defs>${scene.glyphDefs(glyphs)}${glow}</defs>`,
    `<rect width="${width}" height="${height}" fill="${NIGHT.sky}"/>`,
    scene.skyBands({ edges: skyEdges(), lift: plumeLift, tones: skyTones(identity, section.place), width }),
    `<ellipse cx="${fire}" cy="${rim(fire)}" rx="120" ry="40" fill="${heat[1]}" fill-opacity="0.55" `
  + 'filter="url(#glow)"/>',
    `<path d="${scene.trace(width, ridge)}L${width} ${height}L0 ${height}Z" fill="${NIGHT.ridge}"/>`,
    `<path d="${front}" fill="none" stroke="${heat[0]}" stroke-width="2.4"/>`,
    `<g opacity="0.94">${flock}</g>`
  ].join('')
}

/**
 * Lists the edges of the sky's bands from the top down: the four thinner bands
 * stacked above the start, then the four bands of the section's hue splitting
 * the rest of the way to the horizon evenly.
 */
function skyEdges(): number[] {
  const { feather, horizon, start } = SKY
  const top                         = horizon * start
  const edges                       = [0, 1, 2, 3, 4].map(k => top + k / 4 * (horizon - top))
  for (const height of feather.heights) edges.unshift(edges[0]! - height)
  return edges
}

/**
 * Lists the tones of the sky's bands from the top down: the four thinner bands,
 * each its share of the way from the night to the first band of the section's
 * hue, then the four bands of the section's hue, softened toward gray and
 * brightening toward the horizon.
 */
function skyTones(identity: Identity, place: number): string[] {
  const ground        = shadeOf('ground', 'dark', identity)
  const hue           = common.mix(seriesOf(place, 'dark', identity), NIGHT.gray, 0.3)
  const [dim, bright] = [common.mix(hue, ground, 0.82), common.mix(hue, ground, 0.2)]
  const above         = SKY.feather.shares.map(share => common.mix(NIGHT.sky, dim, share)).toReversed()
  return [...above, ...[0, 1, 2, 3].map(k => common.mix(dim, bright, (k / 3) ** 1.1))]
}

/**
 * Lays out the card of a page inside a section: the section's sky, the
 * section's label and place in a tint of its hue over the page's title, and the
 * edition, on the dark scheme's ground.
 */
export function sectionCard(card: SectionCard): layout.Element {
  const { identity, section, title } = card
  const { faces }                    = identity
  const ink                          = shadeOf('ink', 'dark', identity)
  const tint                         = common.mix(seriesOf(section.place, 'dark', identity), ink, 0.4)
  const { size }                     = WORDS.titles.find(({ chars }) => title.length <= chars)!
  const label                        = `${section.label} · ${numeralOf(section.place)}`

  return layout.card({
    backdrop : sky(card),
    column   : { gap: 20, top: WORDS.top, width: WORDS.width },
    edition  : { ...card, opacity: WORDS.edition },

    words: [
      layout.element('div', { alignItems: 'center', display: 'flex', gap: 14 }, [
        layout.element('div', { backgroundColor: tint, height: 2, width: 34 }),
        layout.mono({ color: tint, family: faces.mono.family, size: WORDS.label, text: label })
      ]),
      layout.element('div', { fontFamily: faces.display.family, fontSize: size, lineHeight: 1.04 }, title)
    ]
  })
}
