// @vitest-environment happy-dom
import { expect, it } from 'vitest'

import type { Readout }       from '../../../lib/lookout/drawing.ts'
import { Flying }             from '../../../lib/lookout/flying.ts'
import { LookoutStage }       from '../../../lib/lookout/stage.ts'
import { RELEASED, box, browser } from '../../common/browser.ts'

/**
 * Builds the boxes a Lookout reads into, each an element of its own.
 */
const boxes = (): Readout => ({
  drones       : document.createElement('span'),
  groups       : document.createElement('span'),
  polarization : document.createElement('span'),
  time         : document.createElement('span')
})

it('plays a flight, swaps in another at once, and leaves nothing held once destroyed', () => {
  const stand   = browser()
  const readout = boxes()
  const stage   = new LookoutStage(box('lookout-stage'), new Flying(30, 0), readout)
  stand.see(true)
  stand.frames.run(10, 50)
  expect(readout.drones.textContent).toBe('30')
  expect(Number(readout.time.textContent!.replace(' s', ''))).toBeGreaterThan(0)
  stage.play(new Flying(50, 0))
  expect(readout.drones.textContent).toBe('50')
  expect(readout.time.textContent).toBe('0.0 s')
  stand.frames.run(3)
  const built = stand.built()
  stand.frames.run(30)
  expect(stand.built()).toEqual(built)
  stage.destroy()
  expect(stand.held()).toEqual(RELEASED)
})
