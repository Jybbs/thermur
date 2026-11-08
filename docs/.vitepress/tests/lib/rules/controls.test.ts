// @vitest-environment happy-dom
import { type Mock, describe, expect, it, vi } from 'vitest'

import * as controls from '../../../lib/rules/controls.ts'
import { press }     from '../../common/rules.ts'

/**
 * A group of three radios in the document, the middle one checked at first,
 * beside what it passes each pick to.
 */
function grouped(): { group: HTMLElement, picked: Mock<(value: string) => void> } {
  const picked = vi.fn<(value: string) => void>()
  const group  = controls.radios('b', 'Pointer', [['a', 'One'], ['b', 'Two'], ['c', 'Three']], picked)
  document.body.replaceChildren(group)
  return { group, picked }
}

/**
 * Reads which radios in `group` are checked and which take the focus by tab.
 */
const checked = (group: HTMLElement): string[] =>
  [...group.querySelectorAll<HTMLElement>('[role="radio"]')].map(radio =>
    `${radio.getAttribute('aria-checked')}/${radio.tabIndex}`)

describe('a chip', () => {
  it('sets its name beside an empty value', () => {
    const { element, key, value } = controls.chip('Seventh by rank')
    expect(element.className).toBe('rules-chip')
    expect(key.textContent).toBe('Seventh by rank')
    expect(value.className).toBe('rules-chip-value')
    expect(element.children).toHaveLength(2)
  })
})

describe('a group of radios', () => {
  it('names the group and checks the initial radio alone, the only one a tab reaches', () => {
    const { group, picked } = grouped()
    expect(group.getAttribute('role')).toBe('radiogroup')
    expect(group.getAttribute('aria-label')).toBe('Pointer')
    expect(checked(group)).toEqual(['false/-1', 'true/0', 'false/-1'])
    expect(picked).not.toHaveBeenCalled()
  })

  it('picks the radio pressed, leaving the focus where it was', () => {
    const { group, picked } = grouped()
    group.querySelector<HTMLElement>('[data-value="c"]')!.click()
    expect(picked).toHaveBeenCalledExactlyOnceWith('c')
    expect(checked(group)).toEqual(['false/-1', 'false/-1', 'true/0'])
    expect(document.activeElement).toBe(document.body)
  })

  it.for([
    { from: 'b', key: 'ArrowRight', to: 'c' },
    { from: 'b', key: 'ArrowDown', to: 'c' },
    { from: 'b', key: 'ArrowLeft', to: 'a' },
    { from: 'b', key: 'ArrowUp', to: 'a' },
    { from: 'c', key: 'ArrowRight', to: 'a' },
    { from: 'a', key: 'ArrowLeft', to: 'c' }
  ])('moves the choice from $from to $to on $key, focus and all', ({ from, key, to }) => {
    const { group, picked } = grouped()
    const radio             = group.querySelector(`[data-value="${from}"]`)!
    expect(press(radio, key)).toBe(true)
    expect(picked).toHaveBeenCalledExactlyOnceWith(to)
    expect(document.activeElement).toBe(group.querySelector(`[data-value="${to}"]`))
  })

  it('leaves every other key to the browser', () => {
    const { group, picked } = grouped()
    expect(press(group.querySelector('[data-value="b"]')!, 'Tab')).toBe(false)
    expect(picked).not.toHaveBeenCalled()
  })
})

describe('a slider', () => {
  it('sets a native range named for a screen reader beside its caption, passing on each value', () => {
    const moved = vi.fn<(value: number) => void>()
    const range = { max: 6, min: 1, step: 0.25, value: 2 }
    const label = controls.slider('Range, in bird lengths', range, 'Range', moved)
    const input = label.querySelector('input')!
    expect(label.className).toBe('rules-range')
    expect(label.textContent).toBe('Range')
    expect(input).toMatchObject({ max: '6', min: '1', step: '0.25', type: 'range', value: '2' })
    expect(input.getAttribute('aria-label')).toBe('Range, in bird lengths')
    input.value = '3.5'
    input.dispatchEvent(new Event('input'))
    expect(moved).toHaveBeenCalledExactlyOnceWith(3.5)
  })
})
