import { type Identity, shadeOf } from './identity.ts'
import type { ProjectFacts }      from './manifest.ts'

/**
 * The edition a card names in its lower left corner, its host and version set
 * in the dark scheme's ink at `opacity` beside the small mark.
 */
interface Edition extends Signed {
  opacity : number
}

/**
 * One element of a card as satori takes it, its style written as CSS properties
 * in camel case.
 */
export interface Element {
  props : {
    style     : Style
    children? : (Element | string)[] | string
    height?   : number
    src?      : string
    width?    : number
  }
  type  : 'div' | 'img'
}

/**
 * Words set in capitals by `mono`: `text` in the face `family` at `size` in
 * `color`, with any further `style` laid over.
 */
interface Lettering {
  color  : string
  family : string
  size   : number
  text   : string
  style? : Style
}

/**
 * A card as `card` lays it out: the inner markup of the SVG drawn full bleed
 * beneath it, the words set in a column `column.width` wide at the left inset
 * and `column.top` pixels down with `column.gap` between them, and the edition
 * in its lower left corner.
 */
interface Sheet {
  backdrop : string
  column   : { gap: number, top: number, width: number }
  edition  : Edition
  words    : Element[]
}

/**
 * What signs a card: the host and the version its edition names, the identity
 * it is drawn in, and the dark scheme's small mark as a data address satori
 * reads.
 */
export interface Signed {
  facts    : Pick<ProjectFacts, 'host' | 'version'>
  identity : Identity
  mark     : string
}

/**
 * The style of one element of a card.
 */
type Style = Record<string, number | string>

/**
 * The size of every card, in pixels, the size a link preview reads, and the
 * margin `inset` every card keeps from its edges.
 */
export const CANVAS = { height: 630, inset: 64, width: 1200 } as const
/**
 * The edition every card sets in its lower left corner, in pixels up from the
 * bottom edge: the small mark `mark` wide beside the host and the version set
 * at `size`.
 */
const EDITION = { bottom: 44, mark: 26, size: 16 } as const

/**
 * Lays out the host and the version a card names in its lower left corner,
 * beside the dark scheme's small mark.
 */
function edition({ facts, identity, mark, opacity }: Edition): Element {
  return element('div', {
    alignItems : 'center',
    bottom     : EDITION.bottom,
    display    : 'flex',
    gap        : 12,
    left       : CANVAS.inset,
    position   : 'absolute'
  }, [
    image(mark, EDITION.mark, EDITION.mark),
    mono({
      color  : shadeOf('ink', 'dark', identity),
      family : identity.faces.mono.family,
      size   : EDITION.size,
      style  : { opacity },
      text   : `${facts.host} · ${facts.version}`
    })
  ])
}

/**
 * Lays out a card on the dark scheme's ground, in its ink and the body face:
 * the backdrop drawn as an SVG the card's size, the words in their column, and
 * the edition.
 */
export function card(sheet: Sheet): Element {
  const { identity }             = sheet.edition
  const { height, inset, width } = CANVAS
  const svg                      = '<svg xmlns="http://www.w3.org/2000/svg" '
                                 + `width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`
                                 + `${sheet.backdrop}</svg>`
  return element('div', {
    backgroundColor : shadeOf('ground', 'dark', identity),
    color           : shadeOf('ink', 'dark', identity),
    display         : 'flex',
    fontFamily      : identity.faces.body.family,
    height          : height,
    position        : 'relative',
    width           : width
  }, [
    image(dataAddress(svg, 'image/svg+xml'), width, height, { left: 0, position: 'absolute', top: 0 }),
    element('div', {
      display       : 'flex',
      flexDirection : 'column',
      gap           : sheet.column.gap,
      left          : inset,
      position      : 'absolute',
      top           : sheet.column.top,
      width         : sheet.column.width
    }, sheet.words),
    edition(sheet.edition)
  ])
}

/**
 * Writes `data` as a base64 data address of the media type `type`, the form
 * satori reads an image from.
 */
export const dataAddress = (data: Buffer | string, type: string): string =>
  `data:${type};base64,${Buffer.from(data).toString('base64')}`

/**
 * Builds one element of a card.
 */
export const element = (
  type     : Element['type'],
  style    : Style,
  children : Element['props']['children'] = []
): Element => ({ props: { children, style }, type })

/**
 * Builds an image of a card, `width` by `height` pixels, which satori sizes
 * from these two rather than from the size its file declares.
 */
export const image = (src: string, width: number, height: number, style: Style = {}): Element =>
  ({ props: { height, src, style: { height, width, ...style }, width }, type: 'img' })

/**
 * Sets `text` in capitals, in `color` and the face `family` at `size`, the
 * letters spaced apart by nearly a fifth of the size.
 */
export function mono({ color, family, size, style = {}, text }: Lettering): Element {
  return element('div', {
    color         : color,
    fontFamily    : family,
    fontSize      : size,
    letterSpacing : size * 0.18,
    textTransform : 'uppercase',
    ...style
  }, text)
}
