import { expect, it } from 'vitest'

import { Ring }   from '../../../lib/front/ring.ts'
import { pushed } from '../../common/front.ts'

it('pushes away from the source at full strength once the heat there reaches 0.42, less below it', () => {
  const ring   = new Ring({ fire: 1, radius: 50, x: 0, y: 0 })
  const [x, y] = pushed(ring, 10, 0)
  expect(x).toBeCloseTo(1, 5)
  expect(y).toBeCloseTo(0, 5)
  expect(pushed(ring, 10, 0, 0.21)[0]).toBeCloseTo(0.504, 5)
  expect(Math.hypot(...pushed({ heat: () => 0 }, 10, 0))).toBe(0)
})
