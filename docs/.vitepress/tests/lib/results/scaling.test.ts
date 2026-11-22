import { describe, expect, it } from 'vitest'

import * as scaling from '../../../lib/results/scaling.ts'

/**
 * A statistic measured at three sizes, its control unmeasured at the largest.
 */
const CHART: scaling.Scaling = {
  fixed    : [
    { error: 0.01, mean: 0.5, size: 10 },
    { error: 0.01, mean: 0.4, size: 100 },
    { error: null, mean: null, size: 1000 }
  ],
  flocking : [
    { error: 0.02, mean: 0.9, size: 10 },
    { error: 0.02, mean: 0.8, size: 100 },
    { error: 0.02, mean: 0.7, size: 1000 }
  ],
  label    : 'Share',
  title    : 'Share by Flock Size'
}

describe('the ticks across a range', () => {
  it.for([
    { high: 1, low: 0, want: [0, 0.2, 0.4, 0.6, 0.8, 1] },
    { high: 0.98, low: 0.91, want: [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98] },
    { high: 3000, low: 10, want: [500, 1000, 1500, 2000, 2500, 3000] },
    { high: 2.5, low: -0.5, want: [-0.5, 0, 0.5, 1, 1.5, 2, 2.5] },
    { high: 86, low: -4, want: [0, 20, 40, 60, 80] }
  ])('lays $low to $high out at a round step', ({ high, low, want }) => {
    expect(scaling.ticks(low, high)).toEqual(want)
  })
})

describe('the ticks across a logarithmic range', () => {
  it.for([
    { high: 30, low: 0.15, want: [0.2, 0.5, 1, 2, 5, 10, 20] },
    { high: 4e6, low: 0.1, want: [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000] }
  ])('lays $low to $high out by decade', ({ high, low, want }) => {
    expect(scaling.decades(low, high)).toEqual(want)
  })
})

describe('a figure of a statistic by flock size', () => {
  it('draws a point and a bar per measured size, leaving out a size no seed measured', () => {
    const drawn = scaling.scaling(CHART, 'light')

    expect(drawn.match(/<circle/gu)).toHaveLength(5)
    expect(drawn).toContain('stroke-dasharray="6 4"')
    expect(drawn).toContain('>1,000</text>')
  })

  it.for([
    { norm: { above: false, figure: 0.6, spread: 0.1 }, want: '<rect x="72"' },
    { norm: { above: true, figure: 0.6, spread: null }, want: 'stroke-dasharray="2 4"' }
  ])('draws a norm with a spread as a band and one without as a line', ({ norm, want }) => {
    expect(scaling.scaling({ ...CHART, norm }, 'dark')).toContain(want)
  })

  it('sets a tick in the millions with its suffix', () => {
    const large = CHART.flocking.map(point => ({ ...point, mean: point.mean! * 4e6 }))

    expect(scaling.scaling({ ...CHART, fixed: large, flocking: large }, 'light')).toContain('>3M</text>')
  })

  it('draws a positive statistic spanning decades on a logarithmic axis, keeping small values apart', () => {
    const wide  = [0.2, 1.2, 8e5].map((mean, place) => ({ error: null, mean, size: CHART.flocking[place]!.size }))
    const drawn = scaling.scaling({ ...CHART, fixed: wide, flocking: wide }, 'light')
    const rows  = new Set([...drawn.matchAll(/<circle[^>]* cy="([^"]+)"/gu)].map(match => match[1]))

    expect(rows.size).toBe(3)
    expect(drawn).toContain('>100K</text>')
  })

  it('draws a statistic holding one value throughout', () => {
    const flat = CHART.flocking.map(point => ({ ...point, error: 0, mean: 0 }))

    expect(scaling.scaling({ ...CHART, fixed: flat, flocking: flat }, 'light')).toContain('<svg')
  })
})

describe('the figures The Flock in Still Air embeds', () => {
  it.for(Object.keys(scaling.FIGURES))('writes %s by size', async name => {
    await expect(scaling.scaling(scaling.FIGURES[name]!, 'light'))
      .toMatchFileSnapshot(`../../../../pages/figures/${name}_by_size.svg`)
  })
})
