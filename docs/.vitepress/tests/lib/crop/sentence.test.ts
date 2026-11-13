import { describe, expect, it } from 'vitest'

import { Kept }      from '../../../lib/crop/kept.ts'
import * as sentence from '../../../lib/crop/sentence.ts'

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
