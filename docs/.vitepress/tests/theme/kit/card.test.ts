// @vitest-environment happy-dom
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }     from 'vitest'

import { SOURCES, lineOf, pieceOf }         from '../../../lib/kit/pieces.ts'
import { PieceCard }                        from '../../../theme/kit/card.ts'
import { browser }                          from '../../common/browser.ts'

enableAutoUnmount(afterEach)

afterEach(() => {
  document.documentElement.style.removeProperty('--th-settle')
})

/**
 * The readers a case sets, two modules reading the slider.
 */
const READERS = { slider: ['.vitepress/theme/one.ts', '.vitepress/theme/two.ts'] }

describe('PieceCard', () => {
  it('stacks every piece in one slot, the shown one alone seen and reachable', () => {
    const widget = mount(PieceCard, { props: { readers: READERS, shown: 'slider' } })
    const stack  = widget.get('.kit-card-main .kit-card-stack').findAll(':scope > div')
    expect(stack.filter(one => one.attributes('data-shown') === '').map(one => one.text())).toEqual(['Slider'])
    expect(stack.filter(one => one.attributes('inert') !== undefined)).toHaveLength(stack.length - 1)
    expect(stack.filter(one => one.attributes('aria-hidden') !== 'true').map(one => one.text())).toEqual(['Slider'])
    expect(widget.get('section').attributes()).toMatchObject({ 'aria-label': 'Slider', 'data-part': 'controls' })
  })

  it('names the Zag.js machine a piece runs on, or that it runs on its own', async () => {
    const widget = mount(PieceCard, { props: { readers: {}, shown: 'slider' } })
    expect(widget.get('.kit-card-swatch [data-shown]').text()).toBe('Zag slider')
    await widget.setProps({ shown: 'gate' })
    expect(widget.get('.kit-card-swatch [data-shown]').text()).toBe('Its own')
  })

  it('eases its count of readers to the shown piece’s', async () => {
    const stand  = browser()
    document.documentElement.style.setProperty('--th-settle', '100ms')
    vi.spyOn(performance, 'now').mockReturnValue(0)
    const widget = mount(PieceCard, { props: { readers: READERS, shown: 'slider' } })
    stand.frames.run(12, 16)
    await widget.vm.$nextTick()
    expect(widget.get('.kit-card-figure').text()).toBe('2')
    await widget.setProps({ shown: 'chip' })
    stand.frames.run(12, 16)
    await widget.vm.$nextTick()
    expect(widget.get('.kit-card-figure').text()).toBe('0')
  })
})

describe('PieceCard’s fold', () => {
  it('lists the modules reading the piece, or says none reads it', async () => {
    const widget = mount(PieceCard, { props: { readers: READERS, shown: 'slider' } })
    expect(widget.findAll('[data-shown] .kit-card-paths li').map(li => li.text())).toEqual(READERS.slider)
    await widget.setProps({ shown: 'chip' })
    expect(widget.get('.kit-card-fold [data-shown]').text()).toContain('No module reads it yet.')
  })

  it('sets out each example of the piece at work beside what it does', () => {
    const widget  = mount(PieceCard, { attachTo: document.body, props: { readers: {}, shown: 'radios' } })
    const example = widget.get('.kit-card-fold [data-shown] .kit-card-example')
    expect(example.get('.kit-card-caps').text()).toBe(pieceOf('radios').examples[0]!.name)
    expect(example.find('[role=radiogroup]').exists()).toBe(true)
  })

  it('copies the line taking the piece up', async () => {
    const widget = mount(PieceCard, { attachTo: document.body, props: { readers: {}, shown: 'toggle' } })
    await widget.get('.kit-card-fold [data-shown] .kit-card-fact:last-child .kit-catalog-row button').trigger('click')
    await vi.waitFor(async () => expect(await navigator.clipboard.readText()).toBe(lineOf(pieceOf('toggle'))))
  })

  it('loads the piece’s source once a reader opens it, and once alone', async () => {
    const load   = vi.spyOn(SOURCES, 'motion/timing.ts')
    const widget = mount(PieceCard, { props: { readers: {}, shown: 'timing' } })
    const source = widget.get('.kit-card-fold [data-shown] .kit-card-source')
    expect(source.get('pre').text()).toBe('Loading the source')
    await source.trigger('toggle')
    await flushPromises()
    await vi.waitFor(() => expect(source.get('pre').text()).toContain('export function tween'))
    await source.trigger('toggle')
    expect(load).toHaveBeenCalledOnce()
    expect(source.get('code').text()).toBe('docs/kit/src/motion/timing.ts')
  })
})
