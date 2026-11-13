// @vitest-environment happy-dom
import { Radios, Select, Slider, Toggle }      from '@thermur/kit'
import { enableAutoUnmount, mount }            from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PIECES }   from '../../../lib/kit/pieces.ts'
import { EXAMPLES } from '../../../theme/kit/examples.ts'
import { browser }  from '../../common/browser.ts'

enableAutoUnmount(afterEach)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

/**
 * Mounts the example `name` of the piece `key` into the document.
 */
const shown = (key: string, name: string): ReturnType<typeof mount> => {
  browser()
  return mount(EXAMPLES[key]![name]!, { attachTo: document.body })
}

describe('the examples', () => {
  it('sets out every example each piece’s declaration names, and no other', () => {
    expect(Object.fromEntries(PIECES.map(piece => [piece.key, piece.examples.map(example => example.name)])))
      .toEqual(Object.fromEntries(Object.entries(EXAMPLES).map(([key, named]) => [key, Object.keys(named)])))
  })

  it('steps the chip’s reading on a press, wrapping past the last', async () => {
    const widget = shown('chip', 'A Reading Over a Front')
    const step   = widget.get('button')
    expect(widget.get('.kit-chip').text()).toContain('72 m')
    for (let press = 0; press < 3; press++) (step.element as HTMLElement).click()
    await widget.vm.$nextTick()
    expect(widget.get('.kit-chip').text()).toContain('0 m')
  })
})

describe('the controls’ examples', () => {
  it('holds the wind, the fuel, the seconds, and the press each control sets', async () => {
    const radios = shown('radios', 'Three Winds')
    radios.getComponent(Radios).vm.$emit('change', 'gale')
    await radios.vm.$nextTick()
    expect(radios.getComponent(Radios).props('value')).toBe('gale')
    const select = shown('select', 'Inside a Sentence')
    select.getComponent(Select).vm.$emit('change', 2)
    await select.vm.$nextTick()
    expect(select.getComponent(Select).props('picked')).toBe(2)
    const slider = shown('slider', 'Between Stops')
    slider.getComponent(Slider).vm.$emit('change', 72.4)
    await slider.vm.$nextTick()
    expect(slider.get('.kit-catalog-said').text()).toBe('72 s')
    slider.getComponent(Slider).vm.$emit('commit', 90)
    await slider.vm.$nextTick()
    expect(slider.get('.kit-catalog-said').text()).toBe('90 s')
    const toggle = shown('toggle', 'Play and Hold')
    toggle.getComponent(Toggle).vm.$emit('change', true)
    await toggle.vm.$nextTick()
    expect(toggle.get('.kit-catalog-said').text()).toBe('Pressed')
  })

  it('eases its bar from empty again on each press', async () => {
    const stand  = browser()
    document.documentElement.style.setProperty('--th-settle', '100ms')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    const widget = mount(EXAMPLES.timing!['Ease Again']!, { attachTo: document.body })
    await widget.get('button').trigger('click')
    expect(widget.get('.kit-catalog-bar').attributes('style')).toContain('--eased: 0')
    stand.frames.run(30, 16)
    await widget.vm.$nextTick()
    expect(widget.get('.kit-catalog-bar').attributes('style')).toContain('--eased: 1')
    await widget.get('button').trigger('click')
    widget.unmount()
    expect(stand.held().frames).toBe(0)
  })
})
