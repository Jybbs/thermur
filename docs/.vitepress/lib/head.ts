import type { HeadConfig, TransformContext } from 'vitepress'

import type { Card, Cards }  from './cards.ts'
import { type Identity, type Scheme, fontsHref, shadeOf } from './identity.ts'
import { CANVAS }            from './layout.ts'
import type { ProjectFacts } from './manifest.ts'
import source                from './rendered/identity.json' with { type: 'json' }
import { type Placed, SECTIONS, type Section, sectionOf } from './sections.ts'

/**
 * A page other than the front page as schema.org's `Article`, under its title,
 * its description, its card, and its address, as a part of the site.
 */
interface Article {
  '@type'     : 'Article'
  description : string
  headline    : string
  image       : string
  isPartOf    : WebSite
  url         : string
}

/**
 * The structured data a page served at its own address carries, the site itself
 * on the front page and an article of the site on every other page.
 */
type Linked = { '@context': typeof SCHEMA } & (Article | WebSite & { description: string })

/**
 * What the head's transform reads of the page VitePress is rendering, where the
 * description is the one VitePress writes into the page's meta description, the
 * page's own or else the site's.
 */
type Rendered = Pick<TransformContext, 'description' | 'pageData' | 'siteData'>

/**
 * The site as schema.org's `WebSite`, under its name and the address its front
 * page is served at.
 */
interface WebSite {
  '@type' : 'WebSite'
  name    : string
  url     : string
}

/**
 * The page the site opens on, as VitePress names it beneath the pages' folder.
 */
export const FRONT_PAGE = 'index.md'
/**
 * The vocabulary every page's structured data is written in.
 */
const SCHEMA = 'https://schema.org'
/**
 * The project's name, as the site titles itself.
 */
export const TITLE = 'Thermur'
/**
 * The path in the built site of the main card, which every page outside a
 * section shares, and what its image shows a reader who cannot see it.
 */
const MAIN_CARD = {
  alt  : `The ${TITLE} wordmark and tagline beside a fire seen from above, with a flock flying out of `
       + 'the wordmark to circle it',
  path : 'cards/index.png'
} as const

/**
 * Writes the structured data of a page served at its own address, an article of
 * the site where `article` holds and the site itself on the front page.
 *
 * Args:
 *   article  : Whether the page is any page but the front page.
 *   homepage : The address the site's front page is served at.
 *   page     : The page's description, card, title, and address, beside
 *     the site's name.
 */
function linked(
  article  : boolean,
  homepage : string,
  page     : { description: string, image: string, site: string, title: string, url: string }
): Linked {
  const site = { '@type': 'WebSite', name: page.site } as const
  return article
    ? {
        '@context'  : SCHEMA,
        '@type'     : 'Article',
        description : page.description,
        headline    : page.title,
        image       : page.image,
        isPartOf    : { ...site, url: homepage },
        url         : page.url
      }
    : { '@context': SCHEMA, ...site, description: page.description, url: page.url }
}

/**
 * Writes a meta tag for each entry of `contents`, naming the entry's key under
 * `attribute` beside its content.
 */
function metaTags(attribute: 'name' | 'property', contents: Readonly<Record<string, string>>): HeadConfig[] {
  return Object.entries(contents).map(([key, content]): HeadConfig => ['meta', { content, [attribute]: key }])
}

/**
 * Names the address a page is served at beneath the site's base, with no
 * `.html`, since the site serves clean addresses, and with a folder's overview
 * served at the folder itself.
 */
export const addressOf = (relativePath: string): string =>
  relativePath.replace(/(?:^|(?<=\/))index\.md$|\.md$/u, '')

/**
 * Names the card a page takes, as its path under the built site: the section
 * card beside the page's own address for a page inside a section, and the main
 * card for every other page, the front page and the page met at a missing
 * address among them.
 */
export const cardOf = (relativePath: string, section: Placed | undefined): string =>
  section === undefined ? MAIN_CARD.path : `cards/${relativePath.replace(/\.md$/u, '.png')}`

/**
 * Every page's preview card and the tags that describe the page to a link
 * preview and a search engine, which names each page's card and writes its tags
 * into the page's head as the site renders the page, then writes every card the
 * pages named into the built site once they are all rendered.
 */
export class Previews {
  readonly #cards    : Cards
  readonly #facts    : ProjectFacts
  readonly #named    = new Map<string, Card>()
  readonly #sections : readonly Section[]

  constructor(cards: Cards, facts: ProjectFacts, sections: readonly Section[] = SECTIONS) {
    this.#cards    = cards
    this.#facts    = facts
    this.#sections = sections
  }

  /**
   * Names the card of the page VitePress is rendering and writes the Open Graph
   * and card tags into its head, the page's title standing alone beside the
   * site's name, and holds what a section card shows for the build's end. A
   * page served at its own address also carries a canonical link to that
   * address and its structured data, typed as the site itself on the front page
   * and as an article of the site on every other page, whereas the page met at
   * a missing address carries neither.
   */
  head({ description, pageData, siteData }: Rendered): HeadConfig[] {
    const { homepage }            = this.#facts
    const { relativePath, title } = pageData
    const section                 = sectionOf(relativePath, this.#sections)
    const card                    = cardOf(relativePath, section)
    const image                   = new URL(card, homepage).href
    const address                 = new URL(addressOf(relativePath), homepage).href
    const url                     = pageData.isNotFound === true ? undefined : address
    const article                 = url !== undefined && relativePath !== FRONT_PAGE
    const alt                     = section === undefined
                                  ? MAIN_CARD.alt
                                  : `${title}, in the ${section.label} section, `
                                  + 'over a flock crossing a sky banded above a fire on a ridge'
    if (section !== undefined) this.#named.set(card, { section, title })

    const properties = {
      'og:description'  : description,
      'og:image'        : image,
      'og:image:alt'    : alt,
      'og:image:height' : String(CANVAS.height),
      'og:image:type'   : 'image/png',
      'og:image:width'  : String(CANVAS.width),
      'og:locale'       : siteData.lang.replace('-', '_'),
      'og:title'        : title,
      'og:type'         : article ? 'article' : 'website',
      ...url === undefined ? {} : { 'og:url': url }
    }
    const names = {
      'twitter:card'      : 'summary_large_image',
      'twitter:image'     : image,
      'twitter:image:alt' : alt
    }
    const tags = [...metaTags('property', properties), ...metaTags('name', names)]
    if (url === undefined) return tags

    return [
      ...tags,
      ['link', { href: url, rel: 'canonical' }],
      ['script', { type: 'application/ld+json' }, JSON.stringify(linked(article, homepage, {
        description,
        image,
        site  : siteData.title,
        title : title,
        url
      }))]
    ]
  }

  /**
   * Writes the main card and every section card a page named into `outDir`.
   */
  write(outDir: string): Promise<void> {
    return this.#cards.write(outDir, MAIN_CARD.path, this.#named)
  }
}

/**
 * Writes the head every page shares, meaning the light and the dark scheme's
 * small mark as the favicon, the faces' stylesheet, the site's name for a link
 * preview, and the ground each scheme paints as the browser's own color.
 *
 * Args:
 *   base     : The path the site is served under.
 *   identity : The identity the head reads the marks, the faces, and the
 *     ground from.
 */
export function siteHead(base: string, identity: Identity = source): HeadConfig[] {
  const { marks } = identity
  const schemes   = Object.keys(identity.palette.ground) as Scheme[]

  return [
    ...schemes.map((scheme): HeadConfig => ['link', {
      href  : `${base}${marks.small[scheme]}`,
      media : `(prefers-color-scheme: ${scheme})`,
      rel   : 'icon',
      type  : 'image/svg+xml'
    }]),
    ['link', { href: 'https://fonts.googleapis.com', rel: 'preconnect' }],
    ['link', { crossorigin: '', href: 'https://fonts.gstatic.com', rel: 'preconnect' }],
    ['link', { href: fontsHref(identity), rel: 'stylesheet' }],
    ['meta', { content: TITLE, property: 'og:site_name' }],
    ...schemes.map((scheme): HeadConfig => ['meta', {
      content : shadeOf('ground', scheme, identity),
      media   : `(prefers-color-scheme: ${scheme})`,
      name    : 'theme-color'
    }])
  ]
}
