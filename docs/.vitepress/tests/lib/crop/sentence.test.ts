// @vitest-environment happy-dom
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }       from 'vitest'
import { defineComponent, h }                        from 'vue'

import { Kept }                            from '../../../lib/crop/kept.ts'
import * as sentence                       from '../../../lib/crop/sentence.ts'
import source                              from '../../../lib/rendered/identity.json' with { type: 'json' }
import { type Browser, RELEASED, browser } from '../../common/browser.ts'

/**
 * A mounted sentence holding its height, beside the stand-ins.
 */
interface Held {
  element : HTMLElement
  menus   : HTMLSelectElement[]
  stand   : Browser
  widget  : VueWrapper
}

/**
 * The height the stand-in lays the widest wording of the three menus out at, a
 * pixel for each of its characters.
 */
const WIDEST = 'the whole domain'.length + 'the last two snapshots'.length + '1,000 m'.length

/**
 * Reads a wording's width with the eastern half the widest, though another
 * wording in its menu holds more characters.
 */
const shortWidest = (wording: string): number => wording === 'the eastern half' ? 100 : wording.length

/**
 * Mounts a sentence holding its height into the document.
 */
function held(): Held {
  const stand  = browser()
  const widget = mount(defineComponent({
    setup() {
      const menus = sentence.menusOf(new Kept())
      const words = sentence.useHeldHeight()
      return () => h('p', { ref: words }, menus.map(({ choices }) =>
        h('select', choices.map(({ name }) => h('option', name)))))
    }
  }), { attachTo: document.body })
  const element = widget.element as HTMLElement
  return { element, menus: [...element.querySelectorAll('select')], stand, widget }
}

/**
 * Lays `element` out as the stand-in does, each menu as wide as `widthOf` reads
 * its wording, a pixel for each character unless given, the sentence as high as
 * its menus are wide in all and as wide as `size` holds, and lists the height
 * it held at each measure.
 */
function measuring(
  element : HTMLElement,
  size    : { width: number },
  widthOf : (wording: string) => number = wording => wording.length
): string[] {
  const seen: string[] = []
  const wide           = (menu: HTMLSelectElement): number => widthOf(menu.options[menu.selectedIndex]!.text)
  const menus          = vi.spyOn(HTMLSelectElement.prototype, 'getBoundingClientRect')
  menus.mockImplementation(function measured(this: HTMLSelectElement) {
    return { width: wide(this) } as DOMRect
  })
  vi.spyOn(element, 'getBoundingClientRect').mockImplementation(() => {
    seen.push(element.style.minHeight)
    const height = [...element.querySelectorAll('select')].reduce((all, menu) => all + wide(menu), 0)
    return { height, width: size.width } as DOMRect
  })
  return seen
}

enableAutoUnmount(afterEach)

describe('the menus', () => {
  const menus = sentence.menusOf(new Kept())

  it('name each choice in plain words, each height from the record', () => {
    expect(menus.map(({ choices }) => choices.map(({ name }) => name))).toEqual([
      ['the whole domain', 'the western half', 'the eastern half'],
      ['every snapshot', 'the last two snapshots'],
      ['500 m', '200 m', '1,000 m']
    ])
  })

  it('name themselves for a screen reader', () => {
    expect(menus.map(({ label }) => label)).toEqual(['Where along the ground', 'When', 'How high'])
  })

  it.each([
    sentence.DEFAULTS,
    ...menus.flatMap(({ choices, label }) => [label, ...choices.map(({ name }) => name)])
  ])('read %j as published', text => {
    expect(text).toReadAsPublished()
  })

  it.each([
    { lines: ['x_m = [0.0, 10000.0]'], picks: [1, 0, 0] },
    { lines: ['x_m = [10000.0, 20000.0]'], picks: [2, 0, 0] },
    { lines: ['window_s = [3645.0, 3660.0]'], picks: [0, 1, 0] },
    { lines: ['levels = 10'], picks: [0, 0, 1] },
    { lines: ['levels = 50'], picks: [0, 0, 2] },
    { lines: ['x_m = [10000.0, 20000.0]', 'window_s = [3645.0, 3660.0]', 'levels = 50'], picks: [2, 1, 2] }
  ])('write $lines at the picks $picks under the table’s name', ({ lines, picks }) => {
    expect(sentence.linesOf(menus, picks)).toBe(['[tool.thermur.crop]', ...lines].join('\n'))
  })

  it('write nothing while every pick is a default', () => {
    expect(sentence.linesOf(menus, [0, 0, 0])).toBeNull()
  })
})

describe('a choice of a menu', () => {
  it.each([
    { menu: 0, pick: 1, want: { east: 0.5, first: 0, top: 500, west: 0 } },
    { menu: 0, pick: 2, want: { east: 1, first: 0, top: 500, west: 0.5 } },
    { menu: 1, pick: 1, want: { east: 1, first: 2, top: 500, west: 0 } },
    { menu: 2, pick: 1, want: { east: 1, first: 0, top: 200, west: 0 } },
    { menu: 2, pick: 2, want: { east: 1, first: 0, top: 1000, west: 0 } }
  ])('sets what choice $pick of menu $menu moves, and the first takes it back', ({ menu, pick, want }) => {
    const kept    = new Kept()
    const choices = sentence.menusOf(kept)[menu]!.choices
    const targets = (): Record<string, number> => {
      const { east, first, top, west } = kept
      return { east: east.target, first: first.target, top: top.target, west: west.target }
    }
    choices[pick]!.apply()
    expect(targets()).toEqual(want)
    choices[0]!.apply()
    expect(targets()).toEqual({ east: 1, first: 0, top: 500, west: 0 })
  })
})

describe('the sentence’s height', () => {
  it('holds what the widest wording takes, leaving each menu at its pick', () => {
    const { element, menus, stand } = held()
    measuring(element, { width: 600 })
    menus[0]!.selectedIndex = 2
    stand.reflow(element)
    expect(element.style.minHeight).toBe(`${WIDEST}px`)
    expect(menus.map(({ selectedIndex }) => selectedIndex)).toEqual([2, 0, 0])
  })

  it('reads the widest wording from the layout, whatever its count of characters', () => {
    const { element, stand } = held()
    measuring(element, { width: 600 }, shortWidest)
    stand.reflow(element)
    expect(element.style.minHeight).toBe(`${100 + 'the last two snapshots'.length + '1,000 m'.length}px`)
  })

})

describe('the sentence’s next measure', () => {
  it('clears the height it held before it measures again', () => {
    const { element, stand } = held()
    const size               = { width: 600 }
    const seen               = measuring(element, size)
    stand.reflow(element)
    size.width = 400
    stand.reflow(element)
    expect(seen).toEqual(['', '', `${WIDEST}px`, ''])
  })

  it('measures again once its width changes and at no other time', () => {
    const { element, stand } = held()
    const size               = { width: 600 }
    measuring(element, size)
    stand.reflow(element)
    element.style.minHeight = '1px'
    stand.reflow(element)
    expect(element.style.minHeight).toBe('1px')
    size.width = 400
    stand.reflow(element)
    expect(element.style.minHeight).toBe(`${WIDEST}px`)
  })

  it('measures again once the face it reads in has loaded', async () => {
    const { element, stand } = held()
    measuring(element, { width: 600 })
    expect(stand.fonts.load).toHaveBeenCalledWith(`400 1em '${source.faces.body.family}'`)
    await stand.fonts.load.mock.results[0]!.value
    await Promise.resolve()
    expect(element.style.minHeight).toBe(`${WIDEST}px`)
  })

  it('releases its observer once it unmounts', () => {
    const { stand, widget } = held()
    expect(stand.held().observed).toBe(1)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})
