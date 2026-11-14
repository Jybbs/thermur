// @vitest-environment happy-dom
import * as kit                                      from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }       from 'vitest'

import { fontStack } from '../../../lib/identity.ts'
import source        from '../../../lib/rendered/identity.json' with { type: 'json' }
import * as sketch   from '../../../lib/specimen/sketch.ts'
import IdentityFaces from '../../../theme/identity/faces.ts'

enableAutoUnmount(afterEach)

describe('the rows', () => {
  it('writes a row per face naming it, its weights, its italic, and its stack', () => {
    const rows = mount(IdentityFaces).findAll('.identity-face')
    expect(rows.map(row => row.get('.identity-face-name').text()))
      .toEqual(Object.values(source.faces).map(face => face.family))
    expect(rows[0]!.get('.identity-weights').text()).toBe('400500600700Italic')
    expect(rows[1]!.find('.identity-weights i').exists()).toBe(false)
    expect(rows[2]!.get('.identity-hex').text()).toBe(fontStack(source.faces.mono))
  })

  it('sets each row in its own face and each weight at that weight', () => {
    const rows = mount(IdentityFaces).findAll('.identity-face')
    Object.values(source.faces).forEach((face, place) => {
      expect(rows[place]!.attributes('style')).toEqual(expect.stringContaining(face.family))
      expect(rows[place]!.findAll('.identity-weights span').map(weight => weight.attributes('style')))
        .toEqual(face.weights.map(weight => `font-weight: ${weight};`))
    })
  })

  it('copies a face’s stack on a press', async () => {
    const widget = mount(IdentityFaces, { attachTo: document.body })
    await widget.findAll('.identity-face')[1]!.trigger('click')
    await vi.waitFor(async () =>
      expect(await navigator.clipboard.readText()).toBe(fontStack(source.faces.display)))
  })

  it('holds a face while the focus rests on its row', async () => {
    const widget = mount(IdentityFaces)
    await widget.findAll('.identity-face')[0]!.trigger('focus')
    expect(widget.findAll('.identity-face')[0]!.attributes('data-held')).toBe('')
  })
})

/**
 * Presses `key` on the time slider's thumb, its focus set first.
 */
async function pressed(widget: VueWrapper, key: string): Promise<void> {
  const thumb = widget.get<HTMLElement>('[role=slider]')
  thumb.element.focus()
  await thumb.trigger('keydown', { key })
}

describe('the panel', () => {
  it('picks out a face’s parts of the panel while its row is held, and lets go on leaving', async () => {
    const widget = mount(IdentityFaces)
    await widget.findAll('.identity-face')[2]!.trigger('pointerenter')
    expect(widget.findAll('.identity-dim').map(part => part.attributes('data-face')))
      .toEqual(['display', 'body'])
    expect(widget.findAll('.identity-face')[2]!.attributes('data-held')).toBe('')
    await widget.get('.identity-face-list').trigger('focusout')
    expect(widget.findAll('.identity-dim')).toEqual([])
  })

  it('picks out the face under the pointer across the panel', async () => {
    const widget = mount(IdentityFaces)
    await widget.get('.identity-log').trigger('pointerover')
    expect(widget.findAll('.identity-dim').map(part => part.attributes('data-face')))
      .toEqual(['display', 'mono', 'mono'])
    await widget.get('.identity-panel').trigger('pointerover')
    expect(widget.findAll('.identity-dim')).toEqual([])
    await widget.findAll('.identity-face')[0]!.trigger('pointerenter')
    await widget.get('.identity-panel').trigger('pointerleave')
    expect(widget.findAll('.identity-dim')).toEqual([])
  })

  it('moves the log and settles each gauge to the second the slider reaches', async () => {
    vi.spyOn(kit, 'tween').mockImplementation(step => {
      step(1)
      return () => {}
    })
    const widget = mount(IdentityFaces, { attachTo: document.body })
    await pressed(widget, 'End')
    const reading = sketch.readingsAt(sketch.SPAN)
    expect(widget.findAll('.identity-gauge .identity-reading').map(gauge => gauge.text()))
      .toEqual([reading.altitude, reading.air, reading.split].map(value => String(Math.round(value))))
    expect(widget.get('.identity-log').text()).toBe(sketch.entryAt(sketch.SPAN).slice(1).join(''))
    expect(widget.get('.identity-scrub .identity-reading').text()).toBe(`${sketch.SPAN} s`)
  })

  it('stops a gauge still settling as it unmounts', async () => {
    const stop   = vi.fn<() => void>()
    vi.spyOn(kit, 'tween').mockReturnValue(stop)
    const widget = mount(IdentityFaces, { attachTo: document.body })
    await pressed(widget, 'End')
    await pressed(widget, 'Home')
    widget.unmount()
    expect(stop).toHaveBeenCalledTimes(2)
  })
})

it('labels its readings as made up', () => {
  expect(mount(IdentityFaces).get('figcaption').text()).toBe('A sketch, its readings made up')
})
