import { type Pair, clamp } from '../common.ts'
import { pathThrough }        from '../curve.ts'
import { steps }              from '../scene.ts'

/**
 * The readings a flight's panel shows at a time into the flight, made up for
 * the sketch: the altitude in meters, the air's temperature in kelvin, and the
 * share of the flock split away, in percent.
 */
export interface PanelReadings {
  air      : number
  altitude : number
  split    : number
}

/**
 * The flight chart's size, in the units of its view box.
 */
export const CHART = { height: 200, width: 360 } as const
/**
 * The temperature the flight chart and each spark draw up from, in kelvin.
 */
const FLOOR = 290
/**
 * The temperatures the flight chart marks up its side, in kelvin.
 */
export const KELVINS = [300, 350, 400, 450] as const
/**
 * The temperatures the flight chart rules a line across, in kelvin, made up
 * for the sketch.
 */
export const LEVELS = { critical: 475, warning: 400 } as const
/**
 * The lines of the flight's log, each opening at its second into the flight,
 * its words before the part set in italic, that part, and the words after it.
 */
export const LOG: readonly (readonly [number, string, string, string])[] = [
  [0, 'The flock lifts off the ridge ', 'in one body', '.'],
  [
    41, 'The drones nearest the plume meet its heat ', 'and turn away',
    ', and their neighbors turn with them.'
  ],
  [96, 'The flock splits around the plume ', 'the way a murmuration splits around a falcon', '.'],
  [180, 'Past the plume, the two halves ', 'find each other and close again', '.']
]
/**
 * The heat profile's size, in the units of its view box, the temperatures it
 * spans in kelvin from the air above the plume to the ground beneath it, the
 * meters of height it rises through, and the height in meters its readout
 * reads the air at.
 */
export const PROFILE = { height: 150, kelvins: [300, 470], knot: 30, rise: 300, width: 220 } as const
/**
 * The seconds the flight chart marks along its foot.
 */
export const SECONDS = [0, 60, 120, 180, 240] as const
/**
 * The seconds the sketched flight lasts.
 */
export const SPAN = 240

/**
 * Writes the path of series `series` through the point `placed` sets for each
 * second and temperature, one point every 20 seconds.
 */
const traced = (series: number, placed: (second: number, kelvin: number) => Pair): string =>
  pathThrough(steps(SPAN, 20).map(second => placed(second, temperatureAt(series, second))))

/**
 * Finds the flight chart's distance across for a second into the flight.
 */
export const chartX = (second: number): number => 34 + second / SPAN * (CHART.width - 46)

/**
 * Finds the flight chart's height for a temperature in kelvin.
 */
export const chartY = (kelvin: number): number =>
  CHART.height - 26 - (kelvin - FLOOR) / 210 * (CHART.height - 40)

/**
 * Finds the line of the flight's log in force at `second`.
 */
export const entryAt = (second: number): (typeof LOG)[number] => LOG.findLast(([at]) => at <= second)!

/**
 * Writes the path of the heat profile's fill: the profile, then down its cool
 * edge and back along its foot.
 */
export function profileArea(): string {
  const [cool] = PROFILE.kelvins
  const [x, y] = [profileX(cool), profileY(PROFILE.rise)]
  return `${profilePath()}L${x},${y}L${x},${profileY(0)}Z`
}

/**
 * Writes the path of the air's temperature over height through the plume.
 */
export const profilePath = (): string =>
  pathThrough(steps(PROFILE.rise, 20).map(height => [profileX(warmthAt(height)), profileY(height)]))

/**
 * Finds the heat profile's distance across for a temperature in kelvin.
 */
export function profileX(kelvin: number): number {
  const [cool, hot] = PROFILE.kelvins
  return 30 + (kelvin - cool) / (hot - cool) * (PROFILE.width - 46)
}

/**
 * Finds the heat profile's height for a height in meters.
 */
export const profileY = (height: number): number =>
  PROFILE.height - 22 - height / PROFILE.rise * (PROFILE.height - 34)

/**
 * Reads the flight's panel at `second`.
 */
export const readingsAt = (second: number): PanelReadings => ({
  air      : 300 + 140 * Math.exp(-(((second - 96) / 40) ** 2)),
  altitude : 120 + 40 * Math.sin(second / 40),
  split    : 100 * Math.exp(-(((second - 120) / 46) ** 2))
})

/**
 * Writes the path of series `series` across the flight chart, one point every
 * 20 seconds.
 */
export const seriesPath = (series: number): string =>
  traced(series, (second, kelvin) => [chartX(second), chartY(kelvin)])

/**
 * Writes the path of series `series` as a spark 60 by 20 units, its whole
 * flight across the width.
 */
export const sparkPath = (series: number): string =>
  traced(series, (second, kelvin) => [second / 4, 18 - (kelvin - FLOOR) / 12])

/**
 * Finds the place on a heat ramp of `count` steps that a temperature in kelvin
 * takes, from the first step at the profile's hot end to the last at its cool
 * end.
 */
export function stepAt(kelvin: number, count: number): number {
  const [cool, hot] = PROFILE.kelvins
  return clamp(Math.round((hot - kelvin) / (hot - cool) * (count - 1)), 0, count - 1)
}

/**
 * Finds the temperature series `series` reads `second` into the flight, in
 * kelvin, each series passing the plume a little later, cooler, and wider than
 * the one before it.
 */
export const temperatureAt = (series: number, second: number): number =>
  300 + series * 4
  + 150 * (1 - series * 0.13) * Math.exp(-(((second - 90 - series * 22) / (34 + series * 4)) ** 2))

/**
 * Finds the air's temperature `height` meters over the fire, in kelvin, cooling
 * from the ground's toward the air's above the plume.
 */
export function warmthAt(height: number): number {
  const [cool, hot] = PROFILE.kelvins
  return cool + (hot - cool) * Math.exp(-height / 70)
}
