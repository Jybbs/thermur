// @vitest-environment happy-dom
import { enableAutoUnmount, mount }            from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { fontStack } from '../../../lib/identity.ts'
import source        from '../../../lib/rendered/identity.json' with { type: 'json' }
import * as motion   from '../../../lib/rules/motion.ts'
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

  it('moves the log and settles each gauge to the second the scrubber reaches', async () => {
    vi.spyOn(motion, 'tween').mockImplementation(step => {
      step(1)
      return () => {}
    })
    const widget = mount(IdentityFaces)
    const scrub  = widget.get('input')
    scrub.element.value = '120'
    await scrub.trigger('input')
    const reading = sketch.readingsAt(120)
    expect(widget.findAll('.identity-gauge .identity-reading').map(gauge => gauge.text()))
      .toEqual([reading.altitude, reading.air, reading.split].map(value => String(Math.round(value))))
    expect(widget.get('.identity-log').text()).toBe(sketch.LOG[2]!.slice(1).join(''))
    expect(widget.get('.identity-scrub .identity-reading').text()).toBe('120 s')
  })

  it('stops a gauge still settling as it unmounts', async () => {
    const stop   = vi.fn<() => void>()
    vi.spyOn(motion, 'tween').mockReturnValue(stop)
    const widget = mount(IdentityFaces)
    await widget.get('input').trigger('input')
    await widget.get('input').trigger('input')
    widget.unmount()
    expect(stop).toHaveBeenCalledTimes(2)
  })
})
