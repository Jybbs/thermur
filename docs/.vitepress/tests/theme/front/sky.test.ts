// @vitest-environment happy-dom
import { mount }          from '@vue/test-utils'
import { expect, it, vi } from 'vitest'

import { Flock }             from '../../../lib/front/flock.ts'
import FrontSky              from '../../../theme/front/sky.ts'
import { RELEASED, browser } from '../../common/browser.ts'
import { front }             from '../../common/front.ts'

it('leaves no listener, observer, animation frame, or canvas behind once it unmounts', () => {
  const stand = browser()
  vi.spyOn(Flock.prototype, 'settle').mockImplementation(() => {})
  front()
  const sky = mount(FrontSky, { attachTo: document.body })
  stand.see(true)
  stand.frames.run(3)
  expect(sky.find('canvas').exists()).toBe(true)
  sky.unmount()
  expect(stand.held()).toEqual(RELEASED)
})
