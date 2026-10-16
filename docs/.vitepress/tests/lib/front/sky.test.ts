// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import type { Tones }                from '../../../lib/canvas/tones.ts'
import type { Pair }                 from '../../../lib/common.ts'
import { type Sky, quietAt }         from '../../../lib/front/sky.ts'
import type { Browser }              from '../../common/browser.ts'
import { LAYOUT, geometry, skyOver } from '../../common/front.ts'

/**
 * Flies a fresh sky's flock for half a second with the pointer at `pointer`,
 * then lists the transforms one paint sets.
 */
function flown(pointer: Pair | null): unknown[][] {
  const { context, sky: drawn, stand } = sky()
  for (let k = 0; k < 30; k++) drawn.step(1 / 60, geometry({ pointer }))
  drawn.paint(context, geometry(), 1)
  return transforms(stand)
}

/**
 * Builds a sky fitted to the page `geometry` builds, painting onto the stand-in
 * context, beside the stand-ins themselves.
 */
function sky(): { context: CanvasRenderingContext2D, stand: Browser, sky: Sky, tones: Tones } {
  const { sky: drawn, stand, tones } = skyOver()
  drawn.fit(geometry())
  return { context: stand.pen, sky: drawn, stand, tones }
}

/**
 * Lists every transform a paint set on the context, each bird's among them.
 */
const transforms = (stand: Browser): unknown[][] => stand.context.setTransform!.mock.calls

describe('Sky', () => {
  it.for(['dark', 'light'] as const)('fills the %s night and builds its gradients at first paint', scheme => {
    const { context, sky: drawn, stand, tones } = sky()
    drawn.paint(context, geometry({ scheme }), 1)
    const built = stand.built().gradients
    drawn.paint(context, geometry({ scheme }), 1)
    expect(stand.wrote('fillStyle')).toContain(tones.sky(scheme).top)
    expect(built).toBe(2)
    expect(stand.built().gradients).toBe(built)
  })

  it('sets its horizon at the hero’s foot and its ridge just below it', () => {
    const { sky: drawn } = sky()
    expect(drawn.horizon).toBe(604)
    expect(Math.abs(drawn.ridge(300) - 618)).toBeLessThanOrEqual(14)
  })

  it('sets its fire nearer the middle of a narrow page on a refit', () => {
    const { context, sky: drawn, stand } = sky()
    drawn.fit(geometry({ w: 390 }))
    drawn.paint(context, geometry({ w: 390 }), 1)
    const [, x, , size] = stand.context.drawImage!.mock.calls.at(-1) as number[]
    expect(x! + size! / 2).toBeCloseTo(390 * 0.72)
  })

  it('paints only the fire and the ridge once the night scrolls away, and nothing past the ridge', () => {
    const { context, sky: drawn, stand } = sky()
    drawn.paint(context, geometry({ dy: 700 }), 1)
    expect(stand.context.fill).toHaveBeenCalledOnce()
    expect(stand.context.drawImage).toHaveBeenCalledOnce()
    vi.clearAllMocks()
    drawn.paint(context, geometry({ dy: 1000 }), 1)
    expect(stand.context.fill).not.toHaveBeenCalled()
    expect(stand.context.drawImage).not.toHaveBeenCalled()
  })
})

describe('Sky in flight', () => {
  it('wanders its flock to the far side of the sky once its roost turns, half a minute in', () => {
    const { context, sky: drawn, stand } = sky()
    const across                         = (): number => {
      vi.clearAllMocks()
      drawn.paint(context, geometry(), 1)
      const placed = transforms(stand).filter(([scale]) => scale !== 1)
      return placed.reduce((sum, [, , , , x]) => sum + Number(x), 0) / placed.length
    }
    expect(across()).toBeGreaterThan(600)
    for (let frame = 0; frame < 2100; frame++) drawn.step(1 / 60, geometry())
    expect(across()).toBeLessThan(600)
  })

  it('parts the flock around the pointer over the sky, and pays no heed to it below the ridge', () => {
    const unheeded = flown(null)
    expect(flown([600, 400])).not.toEqual(unheeded)
    expect(flown([600, 700])).toEqual(unheeded)
  })
})

describe('quietAt', () => {
  const page = geometry({ doc: LAYOUT.column })

  it('clears the drawing behind the hero’s words and over the reading column', () => {
    expect(quietAt(page, 600, 320, 0.92)).toBeCloseTo(0.08)
    expect(quietAt(page, 600, 1200, 0.92)).toBeCloseTo(0.3)
    expect(quietAt(page, 40, 1200, 0.92)).toBeCloseTo(1)
  })

  it('keeps the whole drawing beside a page with no reading column', () => {
    expect(quietAt(geometry(), 600, 1200, 0.92)).toBeCloseTo(1)
  })
})
