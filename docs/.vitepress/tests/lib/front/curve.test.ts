import { describe, expect, it } from 'vitest'

import { Curve } from '../../../lib/front/curve.ts'
import parity    from '../../../lib/rendered/parity.json' with { type: 'json' }

describe('Curve', () => {
  it('orders a launch as the package’s zorder does, clipping each window to the ends', () => {
    const { order, points } = parity.curve
    const curve             = new Curve(points.length)
    curve.through(0, new Float64Array(points.flat()), 3)
    expect([...curve.order]).toEqual(order)
    expect(curve.beside(order[3]!, -5)).toBe(order[0])
    expect(curve.beside(order.at(-2)!, 4)).toBe(order.at(-1))
  })
})
