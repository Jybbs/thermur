// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import type { Geometry }                   from '../../../lib/canvas/part.ts'
import { Lines }                           from '../../../lib/front/lines.ts'
import { FIRE, LAYOUT, geometry, skyOver } from '../../common/front.ts'

/**
 * Eight lines fitted to `page` under a fitted sky, painting onto the stand-in
 * context, beside the spy on each bird they draw and the stand-ins.
 */
function lines(page: Geometry = geometry()) {
  const { marks, sky, stand, tones } = skyOver()
  const drawn                        = new Lines(sky, marks, tones, FIRE)
  const birds                        = vi.spyOn(marks, 'bird')
  sky.fit(page)
  drawn.fit(page)
  return { birds, context: stand.pen, drawn, stand, tones }
}

/**
 * Reads the turn of each bird a paint drew.
 */
const turns = (birds: { mock: { calls: unknown[][] } }): unknown[] => birds.mock.calls.map(call => call[5])

describe('Lines', () => {
  it('draws and steps nothing before its first fit', () => {
    const { marks, sky, stand, tones } = skyOver([])
    const drawn                        = new Lines(sky, marks, tones, FIRE)
    drawn.step(1 / 60, geometry())
    drawn.paint(stand.pen, geometry(), 1)
    expect(stand.context.stroke).toBeUndefined()
  })

  it('moves the pen on a spring until it settles where a still page draws it', () => {
    const page                      = geometry({ dy: 1200 })
    const { birds, context, drawn } = lines()
    for (let k = 0; k < 600; k++) drawn.step(1 / 60, page)
    drawn.paint(context, page, 1)
    const settled = turns(birds)
    birds.mockClear()
    drawn.paint(context, { ...page, still: true }, 1)
    expect(turns(birds)).toEqual(settled)
  })

  it('lands the eight level on the ridge at the page’s foot once the page is read to its end', () => {
    const page                      = geometry({ dy: 2200, still: true })
    const { birds, context, drawn } = lines(page)
    drawn.paint(context, page, 1)
    expect(birds).toHaveBeenCalledTimes(8)
    expect(new Set(birds.mock.calls.map(call => call[5]))).toEqual(new Set([0]))
  })

  it('draws the roost at the page’s foot only once it nears the window', () => {
    const near = lines(geometry({ dy: 2200, still: true }))
    near.drawn.paint(near.context, geometry({ dy: 2200, still: true }), 1)
    const far = lines(geometry({ still: true }))
    far.drawn.paint(far.context, geometry({ still: true }), 1)
    expect(near.stand.context.fillRect).toHaveBeenCalled()
    expect(far.stand.context.fillRect).toBeUndefined()
  })

  it('draws the lines over the reading column at less than half their opacity elsewhere', () => {
    const page                             = geometry({ doc: LAYOUT.column, dy: 1200, still: true })
    const { context, drawn, stand, tones } = lines(page)
    drawn.paint(context, page, 1)
    const styles = stand.wrote('strokeStyle')
    expect(styles).toContain(tones.ink('dark', 0.11))
    expect(styles).toContain(tones.ink('dark', 0.11 * 0.45))
    expect(stand.context.clip).toHaveBeenCalledWith('evenodd')
  })
})
