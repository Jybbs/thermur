import { describe, expect, it } from 'vitest'

import source      from '../../../lib/rendered/identity.json' with { type: 'json' }
import { tenths }  from '../../../lib/scene.ts'
import * as sketch from '../../../lib/specimen/sketch.ts'

/**
 * The first point a path is written from.
 */
const opening = (path: string): number[] => /^M([\d.]+),([\d.]+)/u.exec(path)!.slice(1).map(Number)

/**
 * Finds the second series `place` reads its hottest, beside that temperature.
 */
function peakOf(place: number): [number, number] {
  const kelvins = Array.from({ length: sketch.SPAN + 1 }, (_, second) => sketch.temperatureAt(place, second))
  const hottest = Math.max(...kelvins)
  return [kelvins.indexOf(hottest), hottest]
}

describe('the flight chart', () => {
  it('runs the flight across the chart between its side and its right edge', () => {
    expect([sketch.chartX(0), sketch.chartX(sketch.SPAN)]).toEqual([34, sketch.CHART.width - 12])
  })

  it('sets a hotter temperature higher up the chart', () => {
    expect(sketch.chartY(450)).toBeLessThan(sketch.chartY(300))
  })

  it('lays the critical line above the warning line, both inside the chart', () => {
    const [critical, warning] = [sketch.chartY(sketch.LEVELS.critical), sketch.chartY(sketch.LEVELS.warning)]
    expect(critical).toBeLessThan(warning)
    expect(critical).toBeGreaterThan(0)
  })

  it.for(source.palette.series.map((_, place) => place))('starts series %i at its own coolest', place => {
    expect(opening(sketch.seriesPath(place)))
      .toEqual([sketch.chartX(0), tenths(sketch.chartY(sketch.temperatureAt(place, 0)))])
  })

  it('passes each later series through the plume later, cooler, and wider', () => {
    const peaks = source.palette.series.map((_, place) => peakOf(place))
    expect(peaks.map(([second]) => second)).toEqual([90, 112, 134, 156, 178])
    const kelvins = peaks.map(([, kelvin]) => kelvin)
    expect(kelvins).toEqual(kelvins.toSorted((a, b) => b - a))
  })

  it('draws each spark inside its 60 by 20 box', () => {
    const numbers = sketch.sparkPath(0).match(/-?[\d.]+/gu)!.map(Number)
    expect(Math.min(...numbers)).toBeGreaterThanOrEqual(-2)
    expect(Math.max(...numbers.filter((_, place) => place % 2 === 0))).toBeLessThanOrEqual(60)
  })
})

describe('the heat profile', () => {
  it('cools from the ground’s temperature toward the air’s above the plume', () => {
    const [cool, hot] = sketch.PROFILE.kelvins
    expect(sketch.warmthAt(0)).toBe(hot)
    expect(sketch.warmthAt(sketch.PROFILE.rise)).toBeGreaterThan(cool)
    expect(sketch.warmthAt(sketch.PROFILE.rise)).toBeLessThan(cool + 5)
  })

  it('spans its width from the cool end to the hot end, and its height from the ground up', () => {
    const [cool, hot] = sketch.PROFILE.kelvins
    expect([sketch.profileX(cool), sketch.profileX(hot)]).toEqual([30, sketch.PROFILE.width - 16])
    expect(sketch.profileY(sketch.PROFILE.rise)).toBeLessThan(sketch.profileY(0))
  })

  it('closes its fill down the cool edge and back along the foot', () => {
    const [cool] = sketch.PROFILE.kelvins
    const x      = sketch.profileX(cool)
    const top    = sketch.profileY(sketch.PROFILE.rise)
    expect(sketch.profileArea()).toBe(`${sketch.profilePath()}L${x},${top}L${x},${sketch.profileY(0)}Z`)
  })

  it.for([
    [470, 0], [436, 1], [300, 5], [520, 0], [250, 5]
  ])('sets %i K on step %i of six, holding past either end', ([kelvin, place]) => {
    expect(sketch.stepAt(kelvin!, 6)).toBe(place)
  })
})

describe('the flight’s panel', () => {
  it.for([
    [0, 0], [40, 0], [41, 41], [120, 96], [sketch.SPAN, 180]
  ])('reads at second %i the log line opening at %i', ([second, at]) => {
    expect(sketch.entryAt(second!)[0]).toBe(at)
  })

  it('reads the hottest air where the flock meets the plume, and the widest split after it', () => {
    expect(sketch.readingsAt(96).air).toBe(440)
    expect(sketch.readingsAt(120).split).toBe(100)
    expect(sketch.readingsAt(0).altitude).toBe(120)
  })
})
