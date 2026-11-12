import fs from 'node:fs'

import { describe, expect, it } from 'vitest'

import { computed, declared, styled } from '../../common/sheets.ts'

/**
 * Reads the value the tables' sheet declares for `property` in the rule whose
 * selector list is `selector` inside its query for a phone's width.
 */
function narrow(property: string, selector: string): string {
  return styled('tables.css').find(rule =>
    (rule.parentRule as CSSMediaRule | null)?.conditionText === '(width < 40rem)'
    && rule.selectorText.replaceAll(/\s+/gu, ' ') === selector
  )!.style.getPropertyValue(property)
}

describe('a chip in a table', () => {
  it('keeps the first column’s chip whole, since no rule at any width lets it break', () => {
    const wraps = styled('tables.css')
      .filter(rule => rule.selectorText.includes(':first-child code'))
      .map(rule => rule.style.getPropertyValue('overflow-wrap'))
    expect(wraps).toEqual(['normal'])
  })

  it('breaks a chip in any other column at any character', () => {
    expect(declared('overflow-wrap', '.vp-doc td code', 'tables.css')).toBe('anywhere')
  })
})

describe('a table at a phone’s width', () => {
  it('lays each row out as a card of stacked cells', () => {
    expect(narrow('display', '.vp-doc :is(table, tbody, tr, td)')).toBe('block')
    expect(narrow('border-radius', '.vp-doc tr')).toBe('0.5rem')
  })

  it('keeps the columns side by side wider than a phone', () => {
    const page = '<div class="vp-doc"><table id="table"><tbody><tr><td>x</td></tr></tbody></table></div>'
    expect(computed(page, 'display', '#table', 'tables.css')).toBe('table')
  })

  it('opens a gap between two cards and none after the last', () => {
    expect(narrow('margin-top', '.vp-doc tr + tr')).toBe('0.625rem')
    expect(narrow('margin-bottom', '.vp-doc tr')).toBe('')
  })

  it('sets every cell between a row’s first and last on one line', () => {
    expect(narrow('display', '.vp-doc td:not(:first-child, :last-child)')).toBe('inline')
  })

  it('hides the header row from sight while leaving it to label each cell for a screen reader', () => {
    expect(narrow('clip-path', '.vp-doc thead')).toBe('inset(50%)')
    expect(narrow('display', '.vp-doc thead')).toBe('')
  })

  it('separates two values on a line with a dot a screen reader leaves unread', () => {
    const separator = '.vp-doc td:not(:first-child, :last-child, :empty) + td:not(:last-child, :empty)::before'
    expect(narrow('content', separator)).toBe(`' · ' / ''`)
  })

  it('declares the dot without alternative text first, for a browser that drops the form with it', () => {
    const sheet = fs.readFileSync(new URL('../../../theme/styles/tables.css', import.meta.url), 'utf8')
    expect(sheet).toMatch(/content : ' · ';\n\s*content : ' · ' \/ '';/u)
  })
})

describe('the box of a card’s parts at a phone’s width', () => {
  it.for([
    ['position', 'absolute', '.vp-doc thead'],
    ['width', '1px', '.vp-doc thead'],
    ['height', '1px', '.vp-doc thead'],
    ['overflow', 'hidden', '.vp-doc thead'],
    ['white-space', 'nowrap', '.vp-doc thead'],
    ['padding', '0.625rem 0.75rem', '.vp-doc tr'],
    ['padding', '0px', '.vp-doc td'],
    ['border', '0px', '.vp-doc td'],
    ['margin-bottom', '0.375rem', '.vp-doc td:first-child'],
    ['margin-top', '0.25rem', '.vp-doc td:last-child'],
    ['color', 'var(--vp-c-text-2)', '.vp-doc td:last-child'],
    ['color', 'var(--vp-c-text-3)', '.vp-doc td:not(:first-child, :last-child, :empty) + td:not(:last-child, :empty)::before']
  ])('declares %s as %s on %s', ([property, value, selector]) => {
    expect(narrow(property!, selector!)).toBe(value)
  })
})
