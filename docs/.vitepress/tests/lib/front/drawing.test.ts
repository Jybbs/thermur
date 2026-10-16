// @vitest-environment happy-dom
import { expect, it } from 'vitest'

import { drawFront }    from '../../../lib/front/drawing.ts'
import source           from '../../../lib/rendered/identity.json' with { type: 'json' }
import { browser }      from '../../common/browser.ts'
import { FLOCK, front } from '../../common/front.ts'

it('builds no element, gradient, or path on a run of frames after the first fit', () => {
  const stand = browser()
  const stage = drawFront(front(), source, FLOCK)
  stand.see(true)
  stand.frames.run(2)
  const built = stand.built()
  stand.frames.run(30)
  expect(stand.built()).toEqual(built)
  stage.destroy()
})
