// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import { Tones }           from '../../../lib/canvas/tones.ts'
import type { Pair }       from '../../../lib/common.ts'
import { stripY }          from '../../../lib/crop/strip.ts'
import { FIRE }            from '../../../lib/front/drawing.ts'
import { Flock }           from '../../../lib/front/flock.ts'
import { Plume }           from '../../../lib/front/plume.ts'
import { PointerHeat }     from '../../../lib/front/pointer-heat.ts'
import * as crop           from '../../common/crop.ts'
import { FLOCK, geometry } from '../../common/front.ts'

/**
 * The place and the size a rectangle is filled at.
 */
type Box = [number, number, number, number]

/**
 * Flies a fresh strip half a second with the pointer at the place `pointer`
 * reads from its first paint, and reads where its birds end.
 */
function flownUnder(pointer: (strip: crop.Flown) => Pair | null): Float32Array {
  const strip = crop.painted()
  const at    = pointer(strip)
  for (let k = 0; k < 30; k++) strip.part.step(1 / 60, geometry({ ...crop.STRIP, pointer: at }))
  return strip.birds.x
}

describe('stripY', () => {
  it.each([
    { height: 20, want: 125 },
    { height: 200, want: 71.43 },
    { height: 500, want: 37.52 },
    { height: 1000, want: 0 }
  ])('stands $height m $want pixels down the strip, on a square root above the floor', ({ height, want }) => {
    expect(stripY(20, height, 128)).toBeCloseTo(want, 2)
  })
})

describe('the sky a strip flies', () => {
  it('stands 120 pixels tall and as wide as the part at the same scale on both axes', () => {
    const region = vi.spyOn(Flock.prototype, 'region')
    crop.painted()
    expect(region.mock.lastCall![0].y1).toBe(120)
    expect(region.mock.lastCall![0].x1).toBeCloseTo(688 / crop.scaleAt(500), 1)
  })

  it('hands the flock a fresh record for each new width, since the flock keeps the one it receives', () => {
    const region          = vi.spyOn(Flock.prototype, 'region')
    crop.painted(FLOCK, [1, 0, 0])
    const [before, after] = region.mock.calls.map(([record]) => record)
    expect(after).not.toBe(before)
    expect(after!.x1).toBeCloseTo(344 / crop.scaleAt(500), 1)
  })

  it('moves the flock only while the part’s width changes', () => {
    const region   = vi.spyOn(Flock.prototype, 'region')
    const { part } = crop.flown()
    for (let k = 0; k < 5; k++) part.step(1 / 60, geometry(crop.STRIP))
    expect(region).toHaveBeenCalledTimes(1)
  })

  it('carries the front page’s plume at its own scale, below its foot and unseen', () => {
    const heat     = vi.spyOn(Plume.prototype, 'heat')
    const { part } = crop.flown()
    part.step(1 / 60, geometry(crop.STRIP))
    const plume    = heat.mock.contexts.at(-1) as Plume
    expect(plume.fire).toBe(FIRE)
    expect(plume.y).toBeCloseTo(120 + 36 * 120 / 484)
    expect(plume.x).toBeCloseTo(688 / crop.scaleAt(500) * 0.76, 1)
    expect(plume.rise).toMatchObject({ gain: 1.7, lean: 0.42, spread: 0.24 })
    expect(plume.rise.core).toBeCloseTo(22 * 120 / 484)
    expect(plume.rise.reach).toBeCloseTo(300 * 120 / 484)
  })

  it('settles the flock on the first fit alone', () => {
    const settle   = vi.spyOn(Flock.prototype, 'settle')
    const { part } = crop.flown()
    part.fit(geometry(crop.STRIP))
    expect(settle.mock.calls.map(([seconds]) => seconds)).toEqual([FLOCK.settle, 0])
  })
})

describe('the part a strip paints', () => {
  it.each([
    { picks: [0, 0, 0], tall: 87.48, top: 500, west: 0, wide: 688 },
    { picks: [1, 0, 1], tall: 53.57, top: 200, west: 0, wide: 344 },
    { picks: [2, 0, 2], tall: 125, top: 1000, west: 344, wide: 344 }
  ])('tints $wide pixels from $west up to $top m and rules the ground', ({ picks, tall, west, wide }) => {
    const { stand }      = crop.painted(FLOCK, picks)
    const [tint, ground] = stand.context.fillRect!.mock.calls
    expect(tint![0]).toBe(west)
    expect(tint![2]).toBe(wide)
    expect(tint![1] + tint![3]).toBe(125)
    expect(tint![3]).toBeCloseTo(tall, 2)
    expect(ground).toEqual([0, 127, 688, 1])
  })

  it.for(['dark', 'light'] as const)('tints the %s scheme and draws its birds in the ember', scheme => {
    const { bird, part, stand } = crop.flown()
    const tones                 = new Tones()
    part.paint(stand.pen, geometry({ ...crop.STRIP, scheme, still: true }), 1)
    expect(stand.wrote('fillStyle').slice(0, 2)).toEqual([
      tones.flock(scheme, 1, { dark: 0.08, light: 0.09 }[scheme]),
      tones.ink(scheme, 0.22)
    ])
    expect(bird.mock.lastCall![7]).toBe(tones.flock(scheme, 0.85, 1))
  })

  it('glides to a pick while it moves, and takes it at once standing still', () => {
    const { kept, part, stand } = crop.flown(FLOCK, [1, 0, 0])
    part.step(1 / 60, geometry(crop.STRIP))
    expect(kept.east.value).toBeGreaterThan(0.5)
    part.paint(stand.pen, geometry({ ...crop.STRIP, still: true }), 1)
    expect(kept.east.value).toBe(0.5)
  })
})

describe('the gathering of a strip’s birds', () => {
  it.each([
    { picks: [0, 0, 0], tight: 1, top: 500 },
    { picks: [0, 0, 1], tight: 1, top: 200 },
    { picks: [0, 1, 1], tight: Math.SQRT1_2, top: 200 },
    { picks: [1, 1, 1], tight: 0.5, top: 200 }
  ])('closes them to $tight of their spacing at the picks $picks', ({ picks, tight, top }) => {
    const strip            = crop.painted(crop.CROWD, picks)
    const [first, ...rest] = crop.draws(strip)
    const reach            = tight * crop.scaleAt(top)
    expect(rest.length).toBeGreaterThan(0)
    for (const { index, x, y } of rest) {
      expect(x - first!.x).toBeCloseTo((strip.birds.x[index]! - strip.birds.x[first!.index]!) * reach, 2)
      expect(y - first!.y).toBeCloseTo((strip.birds.y[index]! - strip.birds.y[first!.index]!) * reach, 2)
    }
  })

  it('leaves the middle of the birds showing where it stands', () => {
    const strip  = crop.painted(crop.CROWD, [1, 1, 1])
    const shown  = Array.from({ length: 19 }, (_, bird) => Math.min(3, 18.5 - bird) / 3)
    const weight = shown.reduce((total, share) => total + share)
    const middle = shown.reduce((total, share, bird) => total + share * strip.birds.x[bird]!, 0) / weight
    for (const { index, x } of crop.draws(strip)) {
      expect(x / crop.scaleAt(200)).toBeCloseTo(middle + (strip.birds.x[index]! - middle) * 0.5, 2)
    }
  })

  it.each([
    { most: 18, picks: [1, 1, 1] },
    { most: 171, picks: [0, 0, 0] }
  ])('shows them in step with the share kept, up to bird $most at the picks $picks', ({ most, picks }) => {
    const strip = crop.painted(crop.CROWD, picks)
    expect(Math.max(...crop.draws(strip).map(({ index }) => index))).toBeLessThanOrEqual(most)
    expect(strip.stand.wrote('globalAlpha').at(-1)).toBe(1)
  })
})

describe('a strip’s birds at the part', () => {
  it('stay inside it, fading as they near its edge', () => {
    const strip = crop.flown(crop.CROWD)
    for (let k = 0; k < 30; k++) strip.part.step(1 / 30, geometry(crop.STRIP))
    strip.part.paint(strip.stand.pen, geometry(crop.STRIP), 1)
    const [[left, top, wide, tall]] = strip.stand.context.fillRect!.mock.calls as [Box]
    const alphas  = strip.stand.wrote('globalAlpha') as number[]
    const fade    = Math.min(5, tall * 0.12)
    const rooms   = crop.draws(strip).map(({ index, x, y }, drawn) => ({
      alpha : alphas[drawn]!,
      index,
      room  : Math.min(x - left, left + wide - x, y - top, top + tall - y)
    }))
    const nearing = rooms.filter(({ index }) => index < 160).filter(({ room }) => room < fade)
    expect(Math.min(...rooms.map(({ room }) => room))).toBeGreaterThan(0)
    expect(Math.max(...rooms.map(({ alpha }) => alpha))).toBeLessThanOrEqual(0.82)
    expect(nearing.length).toBeGreaterThan(0)
    for (const { alpha, room } of nearing) expect(alpha).toBeCloseTo(0.82 * room / fade)
  })

  it('shrink by the root of the sky’s scale, no shorter than a bird still reads', () => {
    const strip  = crop.painted(crop.CROWD, [1, 1, 1])
    const placed = crop.draws(strip)
    for (const { index, length } of placed) {
      expect(length).toBeCloseTo(Math.max(strip.birds.size[index]! * 0.8 * Math.sqrt(crop.scaleAt(200)), 2.2))
    }
    expect(Math.min(...placed.map(({ length }) => length))).toBeCloseTo(2.2)
  })
})

describe('the pointer over a strip', () => {
  it('reads through the gathering, so a bird stands from the pointer as it stands from the falcon', () => {
    const heat   = vi.spyOn(PointerHeat.prototype, 'heat')
    const strip  = crop.painted(crop.CROWD, [1, 1, 1])
    const placed = crop.draws(strip)
    strip.part.step(0, geometry({ ...crop.STRIP, pointer: [100, 60] }))
    const falcon = heat.mock.contexts.at(-1) as PointerHeat
    const reach  = 0.5 * crop.scaleAt(200)
    expect(falcon.on).toBe(true)
    for (const { index, x, y } of placed) {
      expect(x - 100).toBeCloseTo((strip.birds.x[index]! - falcon.x) * reach, 2)
      expect(y - 60).toBeCloseTo((strip.birds.y[index]! - falcon.y) * reach, 2)
    }
  })

  it('parts the birds around it, and leaves their flight alone off the strip', () => {
    const clear = flownUnder(() => null)
    expect(flownUnder(() => null)).toEqual(clear)
    expect(flownUnder(strip => {
      const { x, y } = crop.draws(strip)[0]!
      return [x, y]
    })).not.toEqual(clear)
  })

  it('turns the falcon off once it leaves the strip', () => {
    const heat     = vi.spyOn(PointerHeat.prototype, 'heat')
    const { part } = crop.flown()
    part.step(1 / 60, geometry({ ...crop.STRIP, pointer: [100, 60] }))
    part.step(1 / 60, geometry(crop.STRIP))
    expect((heat.mock.contexts.at(-1) as PointerHeat).on).toBe(false)
  })
})
