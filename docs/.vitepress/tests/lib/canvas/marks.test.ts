// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import { Marks }   from '../../../lib/canvas/marks.ts'
import { Tones }   from '../../../lib/canvas/tones.ts'
import { browser } from '../../common/browser.ts'

/**
 * Builds a short fire in `scheme` at `strength`, full unless given, at the
 * middle of a window.
 */
const fire = (scheme: 'dark' | 'light', strength = 1) => ({
  fire   : strength,
  half   : 26,
  scheme : scheme,
  x      : 600,
  y      : 400
})

describe('Marks', () => {
  it('builds one path per traced bird and fills the one a bird draws', () => {
    const stand = browser()
    const marks = new Marks(['M0 0Z', 'M1 1Z'], new Tones())
    marks.bird(stand.pen, 2, 1, 10, 20, 0, 9, '#000000')
    marks.bird(stand.pen, 2, 0, 10, 20, 0, 9, '#000000')
    const filled = stand.context.fill!.mock.calls.map(([path]) => path)
    expect(stand.built().paths).toBe(2)
    expect(filled).toHaveLength(2)
    expect(filled[0]).not.toBe(filled[1])
    expect(stand.context.setTransform).toHaveBeenCalledWith(18, 0, -0, 18, 20, 40)
    expect(stand.wrote('fillStyle')).toContain('#000000')
  })

  it('draws the fire’s glow into a canvas of its own once per scheme', () => {
    const stand = browser()
    const marks = new Marks([], new Tones())
    for (const scheme of ['dark', 'light', 'dark', 'light'] as const) marks.fire(stand.pen, fire(scheme))
    expect(stand.built().elements).toBe(2)
  })

  it('draws nothing while the fire is out', () => {
    const stand = browser()
    const marks = new Marks([], new Tones())
    marks.fire(stand.pen, fire('dark', 0))
    expect(stand.context.drawImage).toBeUndefined()
  })
})
