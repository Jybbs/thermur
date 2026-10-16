import fs   from 'node:fs'
import path from 'node:path'

import { Resvg } from '@resvg/resvg-js'
import satori, { type FontStyle, type FontWeight, type SatoriOptions } from 'satori'

import * as layout                       from './layout.ts'
import { mainCard }                      from './main-card.ts'
import * as paths                        from './paths.ts'
import source                            from './rendered/identity.json' with { type: 'json' }
import { glyphsOf }                      from './scene.ts'
import { type SectionCard, sectionCard } from './section-card.ts'

/**
 * What the card of a page inside a section shows, the page's title and the
 * section it sits in.
 */
export type Card = Pick<SectionCard, 'section' | 'title'>

/**
 * What every card is drawn from beside the identity: the host and the version a
 * card names, the faces as satori takes them, the traced birds of the mark as
 * SVG path data, and the dark scheme's small mark and wordmark as data
 * addresses satori reads.
 */
interface Sources {
  facts  : layout.Signed['facts']
  fonts  : SatoriOptions['fonts']
  glyphs : readonly string[]
  images : { small: string, wordmark: string }
}

/**
 * The preview cards a build writes into the built site, each drawn in the
 * identity: a section card for each page inside a section and the main card
 * every other page shares.
 */
export class Cards {
  readonly #identity : layout.Signed['identity']
  readonly #sources  : Sources

  constructor(sources: Sources, identity: layout.Signed['identity']) {
    this.#identity = identity
    this.#sources  = sources
  }

  /**
   * Reads each face the identity names from the file the package ships it as,
   * and the dark scheme's small mark, its wordmark, and the traced birds of its
   * mark from the site's public folder.
   *
   * Args:
   *   metaUrl  : Where the caller sits, from which the repository is found.
   *   facts    : The host the site is served from and the project's version,
   *     which the cards name.
   *   identity : The identity the cards are drawn in.
   */
  static under(
    metaUrl  : string,
    facts    : Sources['facts'],
    identity : layout.Signed['identity'] = source
  ): Cards {
    const { marks } = identity
    const fonts     = Object.values(identity.faces).flatMap(face => face.fonts.map(font => ({
      data   : fs.readFileSync(path.join(paths.fontsDir(metaUrl), font.file)),
      name   : face.family,
      style  : font.style as FontStyle,
      weight : font.weight as FontWeight
    })))
    const read    = (file: string): Buffer => fs.readFileSync(path.join(paths.publicDir(metaUrl), file))
    const mark    = read(marks.mark.dark).toString('utf8')
    const encoded = (file: string, type: string): string => layout.dataAddress(read(file), type)

    return new Cards({
      facts  : facts,
      fonts  : fonts,
      glyphs : glyphsOf(mark),
      images : {
        small    : encoded(marks.small.dark, 'image/svg+xml'),
        wordmark : encoded(marks.wordmark.dark, 'image/png')
      }
    }, identity)
  }

  /**
   * Draws the section card `card` shows as a PNG, or the main card where `card`
   * is not given, through satori into an SVG and through resvg into pixels,
   * loading no face from the machine, since satori draws every letter as a
   * path.
   */
  async drawn(card?: Card): Promise<Buffer> {
    const { height, width } = layout.CANVAS
    const svg               = await satori(this.tree(card), { fonts: this.#sources.fonts, height, width })
    return new Resvg(svg, { font: { loadSystemFonts: false } })
      .render()
      .asPng()
  }

  /**
   * Lays the section card `card` shows out as the elements satori draws, or the
   * main card where `card` is not given.
   */
  tree(card?: Card): layout.Element {
    const { facts, glyphs, images } = this.#sources
    const identity                  = this.#identity
    return card === undefined
      ? mainCard({ facts, glyphs, identity, mark: images.small, wordmark: images.wordmark })
      : sectionCard({ ...card, facts, glyphs, identity, mark: images.small })
  }

  /**
   * Writes the main card and each section card into `outDir` under the path it
   * is named by.
   *
   * Args:
   *   outDir   : The folder the site is built into.
   *   main     : The main card's path in the built site.
   *   sections : What each section card shows, under its path in the built
   *     site.
   */
  async write(outDir: string, main: string, sections: ReadonlyMap<string, Card>): Promise<void> {
    const cards: [string, Card | undefined][] = [[main, undefined], ...sections]
    await Promise.all(cards.map(async ([file, card]) => {
      const target = path.join(outDir, file)
      fs.mkdirSync(path.dirname(target), { recursive: true })
      fs.writeFileSync(target, await this.drawn(card))
    }))
  }
}
