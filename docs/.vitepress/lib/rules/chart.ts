import { seriesProperty } from '../identity.ts'

/**
 * Maps the span `from` onto the span `to` along a straight line.
 */
export const linear = (from: readonly [number, number], to: readonly [number, number]) =>
  (value: number): number => to[0] + (value - from[0]) / (from[1] - from[0]) * (to[1] - to[0])

/**
 * Names the color of the chart series at `place`, the fire's at place 0.
 */
export const series = (place: number): string => `var(${seriesProperty(place)})`
