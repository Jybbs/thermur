// @vitest-environment happy-dom
import { Radios }                                    from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it }           from 'vitest'
import { h }                                         from 'vue'

import { PIECES }            from '../../../lib/kit/pieces.ts'
import KitCatalog            from '../../../theme/kit/catalog.ts'
import { RELEASED, browser } from '../../common/browser.ts'
import { served }            from '../../common/server.ts'

enableAutoUnmount(afterEach)

/**
 * Mounts the catalog into the document over the stand-in browser, the slider
 * read by the one module `readers` names.
 */
function mounted(): { stand: ReturnType<typeof browser>, widget: VueWrapper } {
  const stand  = browser()
  const widget = mount(KitCatalog, { attachTo: document.body, props: { readers: { slider: ['.vitepress/theme/threat.ts'] } } })
  return { stand, widget }
}

/**
 * Reads the head above the rail and the name the card reads out.
 */
const read = (widget: VueWrapper): string[] =>
  [widget.get('.kit-catalog-head .kit-catalog-caps').text(), widget.get('.kit-card-main [data-shown]').text()]

describe('KitCatalog', () => {
  it('renders on the server before any browser stands, opening on the first piece', async () => {
    expect(await served(h(KitCatalog))).toContain(PIECES[0]!.name)
  })

  it('opens on the first piece, its part’s scene on the stage', () => {
    const { widget } = mounted()
    expect(read(widget)).toEqual([`1 of ${PIECES.length} · Chrome`, PIECES[0]!.name])
    expect(widget.find('.kit-catalog-stage [data-piece="chip"]').exists()).toBe(true)
  })

  it('walks Back and Next end to end, setting out each part’s scene as the walk reaches it', async () => {
    const { widget }    = mounted()
    const [back, next]  = widget.findAll('.kit-catalog-head button')
    await back!.trigger('click')
    expect(read(widget)).toEqual([`${PIECES.length} of ${PIECES.length} · Motion`, PIECES.at(-1)!.name])
    expect(widget.find('.kit-catalog-stage [data-piece="gate"]').exists()).toBe(true)
    await next!.trigger('click')
    await next!.trigger('click')
    expect(read(widget)[0]).toBe(`2 of ${PIECES.length} · Chrome`)
  })

  it('places a dot for every piece, a gap between parts, beneath each part’s caption', () => {
    const { widget } = mounted()
    const options    = widget.getComponent(Radios).props('options') as readonly { style: { gridColumn: string }, value: string }[]
    expect(options.map(option => option.value)).toEqual(PIECES.map(piece => piece.key))
    expect(options.slice(0, 3).map(option => option.style.gridColumn)).toEqual(['1', '2', '4'])
    expect(widget.findAll('.kit-catalog-captions span').map(span => span.text())).toEqual(['Chrome', 'Controls', 'Drawing', 'Motion'])
    expect(widget.get('.kit-catalog-captions [data-here]').text()).toBe('Chrome')
  })
})

describe('KitCatalog, pointed at', () => {
  it('walks to the dot a reader picks, and reads out the one a reader points at until the pointer leaves', async () => {
    const { widget } = mounted()
    await widget.getComponent(Radios).vm.$emit('change', 'slider')
    await widget.vm.$nextTick()
    expect(read(widget)).toEqual([`5 of ${PIECES.length} · Controls`, 'Slider'])
    const rail  = widget.get('.kit-catalog-rail')
    const label = widget.findAll('.kit-catalog-dots label').find(one => one.find('input').element.value === 'toggle')!
    await label.trigger('pointerover')
    expect(read(widget)).toEqual([`5 of ${PIECES.length} · Controls`, 'Toggle'])
    await rail.trigger('pointerleave')
    expect(read(widget)[1]).toBe('Slider')
    await rail.trigger('pointerover')
    expect(read(widget)[1]).toBe('Slider')
  })

  it('reads out a piece pointed at on the stage, and walks to one pressed or focused there', async () => {
    const { widget } = mounted()
    const stage      = widget.get('.kit-catalog-stage')
    await stage.get('[data-piece="copy"] button').trigger('pointerover')
    expect(read(widget)).toEqual([`1 of ${PIECES.length} · Chrome`, 'Copy'])
    await stage.trigger('pointerleave')
    await stage.get('[data-piece="copy"] button').trigger('pointerdown')
    expect(read(widget)).toEqual([`2 of ${PIECES.length} · Chrome`, 'Copy'])
    await stage.trigger('pointerover')
    await stage.trigger('focusin')
    expect(read(widget)[0]).toBe(`2 of ${PIECES.length} · Chrome`)
  })
})

describe('KitCatalog’s guide and card', () => {
  it('brackets the walked piece with the guide once the stage measures it', async () => {
    const { stand, widget } = mounted()
    stand.reflow()
    await widget.vm.$nextTick()
    expect(widget.get('.kit-catalog-guide').attributes('data-placed')).toBe('')
    expect(widget.findAll('.kit-catalog-corner')).toHaveLength(4)
  })

  it('counts the modules reading the shown piece, and reads none where the page sets no readers', async () => {
    const { widget } = mounted()
    await widget.getComponent(Radios).vm.$emit('change', 'slider')
    await widget.vm.$nextTick()
    expect(widget.get('.kit-card-count').text()).toBe('1module reads it')
    expect(mount(KitCatalog).get('.kit-card-count').text()).toBe('0modules read it')
  })

  it('releases everything it held once it unmounts', () => {
    const { stand, widget } = mounted()
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})
