import fs   from 'node:fs'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { type Card, Cards } from '../../lib/cards.ts'
import { mix }              from '../../lib/common.ts'
import { colorOf }          from '../../lib/identity.ts'
import * as paths           from '../../lib/paths.ts'
import source               from '../../lib/rendered/identity.json' with { type: 'json' }
import { FACTS }            from '../common/facts.ts'
import { test }             from '../common/repository.ts'

/**
 * One element of a card as the cards lay it out.
 */
type Element = ReturnType<Cards['tree']>

/**
 * A page titled The Alarm in the Methods section, the first section.
 */
const SECTION = { section: { label: 'Methods', place: 1 }, title: 'The Alarm' }
/**
 * The cards, drawn in the faces the package ships and the marks the site's
 * public folder holds.
 */
const cards = Cards.under(import.meta.url, FACTS)
/**
 * The place of each section the palette holds a series color for, past the
 * fire's at place 0.
 */
const places = source.palette.series.slice(1).map((_, index) => index + 1)
/**
 * Each kind of card by name, beside the card that draws it, the main card where
 * none is given.
 */
const KINDS = [['a section card', SECTION], ['the main card', undefined]] as const

/**
 * Finds the element of a card whose only child is `text`.
 */
function holding(node: Element, text: string): Element | undefined {
  const { children } = node.props
  if (children === text) return node
  return Array.isArray(children)
    ? children.flatMap(child => typeof child === 'string' ? [] : [holding(child, text)]).find(Boolean)
    : undefined
}

/**
 * Reports whether a point falls inside the tagline as the main card sets it
 * beneath the wordmark, three lines of 30 pixels across 480.
 */
const inTagline = ([x, y]: [number, number]): boolean => x > 64 && x < 544 && y > 287 && y < 404

/**
 * Reads the places of the traced birds an SVG scene draws.
 */
const placed = (svg: string): [number, number][] =>
  [...svg.matchAll(/<use href="#g\d+"[^>]*translate\(([\d.-]+) ([\d.-]+)\)/gu)]
    .map(([, x, y]) => [Number(x), Number(y)])

/**
 * Reads the width and the height a PNG's header declares.
 */
const size = (png: Buffer): [number, number] => [png.readUInt32BE(16), png.readUInt32BE(20)]

/**
 * Reads the SVG a card's scene is drawn in, out of the data address its first
 * element sets, the main card's where `card` is not given.
 */
function sky(card?: Card): string {
  const [image] = cards.tree(card).props.children as Element[]
  const encoded = image!.props.src!.replace(/^data:image\/svg\+xml;base64,/u, '')
  return Buffer.from(encoded, 'base64').toString('utf8')
}

/**
 * Collects every string a card's elements hold, in the order they are laid out.
 */
const words = (node: Element | string): string[] =>
  typeof node === 'string' ? [node] : [node.props.children ?? []].flat().flatMap(child => words(child))

describe('the main card', () => {
  it('lays out the tagline beside the host and the project’s version', () => {
    expect(words(cards.tree())).toEqual([source.tagline.text, 'example.org · 2.7.1'])
  })

  it('keeps its flock clear of the tagline the card sets beneath the wordmark', () => {
    const birds = placed(sky())
    expect(birds.length).toBeGreaterThan(1000)
    expect(birds.filter(bird => inTagline(bird))).toEqual([])
  })

  test('writes the main card and each section card under the path it is named by', async ({ repository }) => {
    const outDir = repository.root
    vi.spyOn(cards, 'drawn')
      .mockResolvedValueOnce(Buffer.from('main'))
      .mockResolvedValueOnce(Buffer.from('section'))
    await cards.write(outDir, 'cards/index.png', new Map([['cards/methods/alarm.png', SECTION]]))
    expect(fs.readFileSync(path.join(outDir, 'cards', 'index.png'), 'utf8')).toBe('main')
    expect(fs.readFileSync(path.join(outDir, 'cards', 'methods', 'alarm.png'), 'utf8')).toBe('section')
  })
})

describe('a section card', () => {
  it('names the section and its place over the title, beside the host and the project’s version', () => {
    expect(words(cards.tree(SECTION))).toEqual(['Methods · 01', 'The Alarm', 'example.org · 2.7.1'])
  })

  it.for(places)('tints the label of the section at place %i with the series color at that place', place => {
    const card  = cards.tree({ ...SECTION, section: { label: 'Methods', place } })
    const label = holding(card, `Methods · 0${place}`)
    const hue   = colorOf(source.palette.series[place]!.dark)
    expect(label?.props.style.color).toBe(mix(hue, colorOf(source.palette.ink.dark), 0.4))
  })

  it('draws each section’s sky apart from every other section’s', () => {
    const skies = places.map(place => sky({ ...SECTION, section: { label: 'Methods', place } }))
    expect(new Set(skies).size).toBe(places.length)
  })

  it.for([
    ['The Alarm', 88],
    ['How the Flock Breaks', 74],
    ['Obtaining the Plume Simulations', 60]
  ] as const)('sets the title %j at %i pixels', ([title, pixels]) => {
    expect(holding(cards.tree({ ...SECTION, title }), title)?.props.style.fontSize).toBe(pixels)
  })

  it('names a place the palette holds no series color for', () => {
    const place = source.palette.series.length
    expect(() => cards.tree({ ...SECTION, section: { label: 'Methods', place } })).toThrow(
      `the palette holds no series color at place ${place}`
    )
  })
})

it.for(KINDS)('draws %s as a PNG a link preview reads, alike on every run', async ([, card]) => {
  const again           = Cards.under(import.meta.url, FACTS)
  const [first, second] = await Promise.all([cards.drawn(card), again.drawn(card)])
  expect(first.subarray(1, 4).toString()).toBe('PNG')
  expect(size(first)).toEqual([1200, 630])
  expect(first.equals(second)).toBe(true)
})

it.for(KINDS)('lays %s on the dark scheme’s ground in its ink and the body face', ([, card]) => {
  expect(cards.tree(card).props.style).toMatchObject({
    backgroundColor : colorOf(source.palette.ground.dark),
    color           : colorOf(source.palette.ink.dark),
    fontFamily      : source.faces.body.family
  })
})

it.for(KINDS)(
  'defines each traced bird of the dark scheme’s mark in %s',
  ([, card]) => {
    const mark = fs.readFileSync(path.join(paths.publicDir(import.meta.url), source.marks.mark.dark), 'utf8')
    expect(sky(card).match(/<path id="g\d+"/gu)).toHaveLength(mark.match(/<path\b/gu)!.length)
  }
)
