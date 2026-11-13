// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import { type Option, Radios } from '../../src/controls/radios.tsx'

/**
 * The radios these cases mount, the last placed over a drawing.
 */
const OPTIONS = [
  { title: 'The first', value: 'a', words: 'Alpha' },
  { value: 'b', words: 'Beta' },
  { style: { left: '4px', top: '8px' }, value: 'c', words: 'Gamma' }
]

/**
 * Mounts the radios with the one at `value` on.
 */
const radios = (value: string) =>
  mount(Radios, {
    attachTo : document.body,
    props    : { label: 'Pointer', options: OPTIONS, value },
    slots    : { item: (option?: Option) => `[${option?.words[0]}]` }
  })

enableAutoUnmount(afterEach)

describe('Radios', () => {
  it('names its group from its label, each radio its words, the one on checked', () => {
    const widget = radios('b')
    const group  = widget.get('[role=radiogroup]')
    expect(document.querySelector(`[id="${group.attributes('aria-labelledby')!}"]`)?.textContent).toBe('Pointer')
    expect(widget.findAll('input[type=radio]').map(input => (input.element as HTMLInputElement).checked))
      .toEqual([false, true, false])
    expect(widget.findAll('.kit-radios-item').map(item => item.text())).toEqual(['[A]Alpha', '[B]Beta', '[G]Gamma'])
  })

  it('carries each radio’s title and its place over a drawing', () => {
    const items = radios('a').findAll('.kit-radios-item')
    expect(items[0]!.attributes('title')).toBe('The first')
    expect((items[2]!.element as HTMLElement).style.left).toBe('4px')
  })

  it('lays its radios along a row, so the thumb slides by its left edge', () => {
    const widget = radios('a')
    expect(widget.get('[role=radiogroup]').attributes('aria-orientation')).toBe('horizontal')
    expect((widget.get('.kit-radios-thumb').element as HTMLElement).style.left).toBe('var(--left)')
  })

  it('reports a pick and shows whatever its caller passes back', async () => {
    const widget = radios('a')
    await widget.findAll('input[type=radio]')[2]!.trigger('click')
    expect(widget.emitted('change')).toEqual([['c']])
    await widget.setProps({ value: 'c' })
    expect(widget.findAll('.kit-radios-item')[2]!.attributes('data-state')).toBe('checked')
  })
})
