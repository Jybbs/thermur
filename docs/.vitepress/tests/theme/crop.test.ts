// @vitest-environment happy-dom
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }       from 'vitest'
import { createSSRApp }                              from 'vue'
import { renderToString }                            from 'vue/server-renderer'

import { Flock }                           from '../../lib/front/flock.ts'
import CropKept                            from '../../theme/crop.ts'
import { type Browser, RELEASED, browser } from '../common/browser.ts'

/**
 * A mounted widget over the stand-in browser, beside the stand-ins.
 */
interface Mounted {
  stand  : Browser
  widget : VueWrapper
}

/**
 * The default crop's words, each menu standing at its first choice.
 */
const DEFAULTS = 'These are the defaults, so there is nothing to set.'

/**
 * Opens menu `index` and picks the choice named `name` from its list.
 */
async function pick(widget: VueWrapper, index: number, name: string): Promise<void> {
  const menu = widget.findAll('.kit-select')[index]!
  await menu.get('[role=combobox]').trigger('click')
  await menu.findAll('[role=option]').find(option => option.text() === name)!.trigger('click')
}

/**
 * Reads the choice each menu shows.
 */
const shown = (widget: VueWrapper): string[] =>
  widget.findAll('.kit-select-words [data-picked]').map(choice => choice.text())

/**
 * Mounts the widget into the document with the frames running, or standing
 * still for a reader who prefers reduced motion, its flock left unsettled.
 */
function mounted(reduced = false): Mounted {
  const stand  = browser(reduced)
  vi.spyOn(Flock.prototype, 'settle').mockImplementation(() => {})
  const widget = mount(CropKept, { attachTo: document.body })
  stand.see(true)
  return { stand, widget }
}

enableAutoUnmount(afterEach)

afterEach(() => {
  delete document.documentElement.dataset.paused
})

describe('the widget’s markup', () => {
  it('reads as one sentence of three named menus over the record’s floor', () => {
    const { widget } = mounted(true)
    const labels     = widget.findAll('[role=combobox]').map(menu =>
      document.querySelector(`[id="${menu.attributes('aria-labelledby')!}"]`)?.textContent)
    expect(labels).toEqual(['Where along the ground', 'When', 'How high'])
    expect(shown(widget)).toEqual(['the whole domain', 'every snapshot', '500 m'])
    expect(widget.get('.crop-sentence').element.childNodes[4]?.textContent).toBe(', from 20 m up to ')
    expect(widget.get('.crop-sentence').text()).toMatch(/^Read .* for .*, from 20 m up to .*\.$/su)
  })

  it('hides its sketch from a screen reader, which reads the sentence instead', () => {
    const { widget } = mounted(true)
    expect(widget.get('.crop-canvas').attributes('aria-hidden')).toBe('true')
    expect(widget.find('.crop-canvas canvas').exists()).toBe(true)
  })

  it('announces its lines as they change, opening on the defaults', () => {
    const { widget } = mounted(true)
    const lines      = widget.get('.crop-lines')
    expect(lines.attributes('aria-live')).toBe('polite')
    expect(lines.attributes('data-empty')).toBe('')
    expect(lines.text()).toBe(DEFAULTS)
  })

  it('renders its words and menus on the server, with no canvas or observer', async () => {
    const html = await renderToString(createSSRApp(CropKept))
    expect(html).toContain('How high')
    expect(html).toContain(DEFAULTS)
    expect(html).not.toContain('<canvas')
  })
})

describe('a pick', () => {
  it('writes its line under the table’s name, and clears it back at a default', async () => {
    const { widget } = mounted(true)
    await pick(widget, 0, 'the western half')
    const lines = widget.get('.crop-lines')
    expect(lines.element.textContent).toBe('[tool.thermur.crop]\nx_m = [0.0, 10000.0]')
    expect(lines.attributes('data-empty')).toBeUndefined()
    await pick(widget, 0, 'the whole domain')
    expect(lines.text()).toBe(DEFAULTS)
    expect(lines.attributes('data-empty')).toBe('')
  })

  it('writes one line for each menu off its default', async () => {
    const { widget } = mounted(true)
    await pick(widget, 0, 'the eastern half')
    await pick(widget, 1, 'the last two snapshots')
    await pick(widget, 2, '1,000 m')
    expect(widget.get('.crop-lines').element.textContent).toBe([
      '[tool.thermur.crop]',
      'x_m = [10000.0, 20000.0]',
      'window_s = [3645.0, 3660.0]',
      'levels = 50'
    ].join('\n'))
  })

  it('repaints a drawing standing still at once', async () => {
    const { stand, widget } = mounted(true)
    await pick(widget, 0, 'the eastern half')
    const [west, , wide] = stand.context.fillRect!.mock.calls.at(-2)!
    expect(west).toBe(600)
    expect(wide).toBe(600)
  })

  it('leaves each menu at the pick it holds across a repaint', async () => {
    const { widget } = mounted(true)
    await pick(widget, 2, '200 m')
    await pick(widget, 0, 'the western half')
    expect(shown(widget)).toEqual(['the western half', 'every snapshot', '200 m'])
  })
})

describe('the widget in motion', () => {
  it('leaves no listener, observer, animation frame, or canvas behind once it unmounts', () => {
    const { stand, widget } = mounted()
    stand.frames.run(3)
    expect(stand.held().canvases).toBe(1)
    widget.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })

  it('stands still while the page’s pause is on, and runs again once it is off', () => {
    const { stand } = mounted()
    stand.frames.run(2)
    document.documentElement.dataset.paused = 'true'
    stand.mutate()
    stand.frames.run(2)
    expect(stand.held().frames).toBe(0)
    document.documentElement.dataset.paused = 'false'
    stand.mutate()
    expect(stand.held().frames).toBe(1)
  })

  it('stands still for a reader who prefers reduced motion, having painted one frame', () => {
    const { stand } = mounted(true)
    expect(stand.held().frames).toBe(0)
    expect(stand.context.fillRect).toHaveBeenCalled()
  })
})
