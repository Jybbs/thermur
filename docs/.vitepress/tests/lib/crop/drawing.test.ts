// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'

import type { Stage } from '../../../lib/canvas/stage.ts'
import { drawKept }   from '../../../lib/crop/drawing.ts'
import { Kept }       from '../../../lib/crop/kept.ts'
import * as harness   from '../../common/browser.ts'
import { FLOCK }      from '../../common/front.ts'

/**
 * Draws a strip over the stand-in browser into a box of the stand-in's size,
 * seen, beside the stand-ins.
 */
function drawn(reduced = false): { stage: Stage, stand: harness.Browser } {
  const stand = harness.browser(reduced)
  const stage = drawKept(harness.box('crop-canvas'), new Kept(), undefined, FLOCK)
  stand.see(true)
  return { stage, stand }
}

describe('drawKept', () => {
  it('builds no element, gradient, or path across a hundred frames once fitted and painted', () => {
    const { stand } = drawn()
    stand.frames.run(2)
    const built = stand.built()
    stand.frames.run(100)
    expect(stand.built()).toEqual(built)
  })

  it('stands still for a reader who prefers reduced motion, having painted one frame', () => {
    const { stand } = drawn(true)
    expect(stand.held().frames).toBe(0)
    expect(stand.context.fillRect).toHaveBeenCalled()
  })

  it('leaves no listener, observer, animation frame, or canvas behind once it stops', () => {
    const { stage, stand } = drawn()
    stand.frames.run(3)
    stage.destroy()
    expect(stand.held()).toEqual(harness.RELEASED)
  })
})
