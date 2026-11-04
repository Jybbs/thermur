// @vitest-environment happy-dom
import { wcagContrast }         from 'culori'
import { describe, expect, it } from 'vitest'

import { Tones }                      from '../../../lib/canvas/tones.ts'
import { shadeOf }                    from '../../../lib/identity.ts'
import { drawThreat }                 from '../../../lib/threat/drawing.ts'
import { Front }                      from '../../../lib/threat/front.ts'
import { Playback }                   from '../../../lib/threat/playback.ts'
import type { Readout }               from '../../../lib/threat/readings.ts'
import { visible }                    from '../../../lib/threat/shades.ts'
import { type Browser, box, browser } from '../../common/browser.ts'

/**
 * A drawing of the front over the stand-in browser, beside its clock, the
 * stand-ins, and the elements it writes into.
 */
interface Drawn {
  playback : Playback
  readout  : Readout
  stand    : Browser
}

/**
 * Draws the front over the stand-in browser into a canvas of the stand-in's
 * size, seen or not.
 */
function drawn(reduced = false, seen = true): Drawn {
  const stand    = browser(reduced)
  const host     = box('threat-canvas')
  const front    = new Front()
  const playback = new Playback(front)
  const readout  = {
    first : document.createElement('span'),
    front : document.createElement('span'),
    last  : document.createElement('span'),
    rail  : document.createElement('input'),
    time  : document.createElement('span')
  }
  drawThreat(front, host, playback, readout)
  stand.see(seen)
  return { playback, readout, stand }
}

/**
 * Lists the words the stand-in context wrote since it last cleared.
 */
const words = (stand: Browser): unknown[] => stand.context.fillText!.mock.calls.map(([word]) => word)

describe('the heat ramp a plot draws in', () => {
  const tones = new Tones()

  it.for(['dark', 'light'] as const)('holds a mark’s contrast on the %s ground, cool to hot', scheme => {
    const ramp = visible(tones.heat, scheme)
    expect(ramp).toHaveLength(3)
    for (const color of ramp) expect(wcagContrast(color, shadeOf('ground', scheme))).toBeGreaterThanOrEqual(3)
  })

  it('runs the dark ground’s to the hottest of the ramp', () => {
    expect(visible(tones.heat, 'dark').at(-1)).toBe(tones.heat[0])
  })
})

describe('a plot in motion', () => {
  it('builds no element, gradient, or path across a hundred frames once fitted and painted', () => {
    const { stand } = drawn()
    stand.frames.run(2)
    const built = stand.built()
    stand.frames.run(100)
    expect(stand.built()).toEqual(built)
  })

  it('writes the time, the front, and the air into its chips as the clock runs', () => {
    const { readout, stand } = drawn()
    stand.frames.run(100, 50)
    expect(readout.time.textContent).toBe('50 s')
    expect(readout.front.textContent).toBe('50 m')
    expect(readout.last.textContent).toBe('299.9 K')
    expect(readout.rail.getAttribute('aria-valuetext')).toBe('50 seconds, the front 50 meters east')
    expect(readout.rail.value).toBe('50')
  })

  it('writes a chip only as its value changes', () => {
    const { playback, readout, stand } = drawn()
    playback.press()
    playback.seek(60)
    stand.frames.run(300, 16)
    const nodes = [readout.first.firstChild, readout.time.firstChild]
    stand.frames.run(10)
    expect([readout.first.firstChild, readout.time.firstChild]).toEqual(nodes)
    expect(readout.first.firstChild).toBe(nodes[0])
    expect(readout.time.firstChild).toBe(nodes[1])
  })

  it('draws the fire’s glow once the front has left the western edge, and none before', () => {
    const { stand } = drawn()
    expect(stand.context.drawImage).toBeUndefined()
    stand.frames.run(1, 16)
    expect(stand.context.drawImage).toHaveBeenCalledOnce()
  })

  it('strokes a trace for each snapshot of the last run while that fades', () => {
    const { playback, stand } = drawn()
    playback.seek(120)
    while (playback.fading === 0) stand.frames.run(1, 16)
    const before = stand.context.stroke!.mock.calls.length
    stand.frames.run(1, 16)
    expect(stand.context.stroke!.mock.calls.length - before).toBeGreaterThan(9)
  })
})

describe('a plot standing still', () => {
  it('shows the middle of the run for a reader who prefers reduced motion', () => {
    const { readout, stand } = drawn(true)
    expect(readout.time.textContent).toBe('60 s')
    expect(stand.held().frames).toBe(0)
  })
})

describe('a plot’s fit', () => {
  it('writes the longest name of the distance axis that fits, and the shorter past that', () => {
    const { stand } = drawn(true)
    stand.context.measureText!.mockImplementation((word: string) => ({
      actualBoundingBoxLeft  : 0,
      actualBoundingBoxRight : word.length * 8
    }))
    stand.resize(500, 300)
    stand.reflow()
    expect(words(stand)).toContain('Distance east of the western edge (m)')
    stand.context.fillText!.mockClear()
    stand.resize(200, 300)
    stand.reflow()
    expect(words(stand)).toContain('Distance east (m)')
    expect(words(stand)).not.toContain('Distance east of the western edge (m)')
  })

  it('falls back on the shortest name where none fits', () => {
    const { stand } = drawn(true)
    stand.context.measureText!.mockReturnValue({ actualBoundingBoxLeft: 0, actualBoundingBoxRight: 9e3 })
    stand.context.fillText!.mockClear()
    stand.reflow()
    expect(words(stand)).toContain('Distance east (m)')
  })

  it('lines the rail’s control up with the rail drawn', () => {
    const { readout } = drawn(true)
    expect(readout.rail.style.getPropertyValue('--threat-rail-left')).toBe('34px')
    expect(readout.rail.style.getPropertyValue('--threat-rail-right')).toBe('10px')
  })

})

describe('a plot’s colors and faces', () => {
  it('fits again once the body’s face has loaded', async () => {
    const { stand } = drawn(true)
    expect(stand.fonts.load).toHaveBeenCalledWith('400 1em \'Crimson Pro\'')
    stand.context.createLinearGradient!.mockClear()
    await Promise.resolve()
    await Promise.resolve()
    expect(stand.context.createLinearGradient).toHaveBeenCalled()
  })

  it('builds each scheme’s colors once, again only after a fit', () => {
    const { stand } = drawn(true)
    const { gradients } = stand.built()
    document.documentElement.classList.add('dark')
    stand.mutate()
    const dark = stand.built().gradients
    expect(dark - gradients).toBeGreaterThan(0)
    document.documentElement.classList.remove('dark')
    stand.mutate()
    document.documentElement.classList.add('dark')
    stand.mutate()
    expect(stand.built().gradients).toBe(dark)
    document.documentElement.classList.remove('dark')
  })
})
