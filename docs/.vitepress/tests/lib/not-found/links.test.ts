import { describe, expect, it, vi } from 'vitest'

import { Links } from '../../../lib/not-found/links.ts'

/**
 * Twenty birds in a row ten pixels apart, the first of them picked, so its
 * nearest neighbors are the next birds along the row in order.
 */
function row(): { xs: Float32Array, ys: Float32Array } {
  return { xs: Float32Array.from({ length: 20 }, (_, index) => index * 10), ys: new Float32Array(20) }
}

/**
 * Moves `links` on by `seconds` at sixty frames a second, the first bird of the
 * row picked throughout.
 */
function run(links: Links, birds: ReturnType<typeof row>, seconds: number, out = true): void {
  for (let frame = 0; frame < seconds * 60; frame++) {
    links.pick(0, birds.xs, birds.ys, 1 / 60, false)
    links.reachOn(1 / 60, out, false)
  }
}

/**
 * Counts the lines `links` draws in one paint.
 */
function strokes(links: Links, birds: ReturnType<typeof row>): number {
  const ctx = {
    beginPath : vi.fn<() => void>(),
    lineTo    : vi.fn<() => void>(),
    moveTo    : vi.fn<() => void>(),
    stroke    : vi.fn<() => void>()
  }
  links.paint(ctx as unknown as CanvasRenderingContext2D, birds.xs, birds.ys)
  return ctx.stroke.mock.calls.length
}

/**
 * Reads how warm each bird of the row the links make it.
 */
function warmth(links: Links): number[] {
  const glow = new Float32Array(20)
  links.glow(glow)
  return [...glow].map(value => Math.round(value * 100) / 100)
}

describe('Links reaching out and drawing back', () => {
  it('reaches out to the seven nearest one after another, nearest first', () => {
    const [links, birds] = [new Links(), row()]
    run(links, birds, 0.2)
    const early = warmth(links)
    expect(early[1]).toBeGreaterThan(early[2]!)
    expect(early[7]).toBe(0)
    run(links, birds, 1)
    const seven = Array.from({ length: 7 }, () => 0.8)
    expect(warmth(links)).toEqual([0, ...seven, ...Array.from({ length: 12 }, () => 0)])
    expect(strokes(links, birds)).toBe(7)
  })

  it('draws every link back once the links hide, letting each bird go', () => {
    const [links, birds] = [new Links(), row()]
    run(links, birds, 1)
    run(links, birds, 0.25, false)
    expect(strokes(links, birds)).toBe(0)
    expect(warmth(links).every(value => value === 0)).toBe(true)
  })

  it('draws a link back before it reaches for the bird that took its fallen bird’s place', () => {
    const [links, birds] = [new Links(), row()]
    run(links, birds, 1)
    birds.xs[3] = 10_000
    run(links, birds, 0.1)
    expect(warmth(links)[8]).toBe(0)
    expect(warmth(links)[3]).toBeGreaterThan(0)
    run(links, birds, 0.6)
    expect(warmth(links)[8]).toBe(0.8)
    expect(warmth(links)[3]).toBe(0)
  })

  it('keeps a link while its bird stays among the nine nearest', () => {
    const [links, birds] = [new Links(), row()]
    run(links, birds, 1)
    birds.xs[7] = 85
    birds.xs[8] = 70
    run(links, birds, 0.1)
    expect(warmth(links)[7]).toBe(0.8)
  })
})

describe('Links standing still, following, and resetting', () => {
  it('shows every link fully out or fully home at once while the drawing stands still', () => {
    const [links, birds] = [new Links(), row()]
    links.pick(0, birds.xs, birds.ys, 1, true)
    links.reachOn(1, true, true)
    expect(strokes(links, birds)).toBe(7)
    links.reachOn(1, false, true)
    expect(strokes(links, birds)).toBe(0)
  })

  it('brings every link still reaching out fully out the moment the drawing stands still', () => {
    const [links, birds] = [new Links(), row()]
    run(links, birds, 0.3)
    expect(warmth(links).slice(1, 8)).not.toEqual(Array.from({ length: 7 }, () => 0.8))
    links.reachOn(0.001, true, true)
    expect(warmth(links).slice(1, 8)).toEqual(Array.from({ length: 7 }, () => 0.8))
  })

  it('starts the links at the picked bird and follows it without overshooting', () => {
    const [links, birds] = [new Links(), row()]
    links.pick(0, birds.xs, birds.ys, 1 / 60, false)
    expect([links.anchor.x, links.anchor.y]).toEqual([0, 0])
    links.fade(1, true, true)
    birds.xs[0]           = 50
    const trail: number[] = []
    for (let frame = 0; frame < 30; frame++) {
      links.pick(0, birds.xs, birds.ys, 0.05, false)
      trail.push(links.anchor.x)
    }
    expect(trail.toSorted((a, b) => a - b)).toEqual(trail)
    expect(Math.max(...trail)).toBeLessThanOrEqual(50)
    expect(trail.at(-1)).toBeCloseTo(50, 0)
  })

  it('fades the links in and out over time, at once while the drawing stands still', () => {
    const links = new Links()
    links.fade(0.1, true, false)
    expect(links.anchor.w).toBeCloseTo(1 - Math.exp(-0.7), 6)
    links.fade(0.1, false, true)
    expect(links.anchor.w).toBe(0)
  })

  it('lets every link go once reset', () => {
    const [links, birds] = [new Links(), row()]
    run(links, birds, 1)
    links.reset()
    expect(strokes(links, birds)).toBe(0)
  })
})
