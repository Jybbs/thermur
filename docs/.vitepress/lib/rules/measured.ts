import trials     from '../measured/trials.json' with { type: 'json' }
import conditions from '../rendered/trials.json' with { type: 'json' }
import { series } from './chart.ts'

/**
 * A rule's name in full and the shorter name a narrow chart sets, by the name
 * of the candidate the trial flew, the default's first.
 */
export type Names = Readonly<Record<string, readonly [string, string]>>
/**
 * One condition of a trial as a chart reads it: what the condition sets, and
 * the statistic's mean across seeds beside its standard error.
 */
export interface Point {
  error : number
  given : number
  mean  : number
}
/**
 * One rule as a chart draws it: its color, its name in full and short, and
 * its points in order of what each condition sets.
 */
export interface Rule {
  color  : string
  label  : string
  points : Point[]
  short  : string
}
/**
 * What one trial measured: what each condition sets, as `mise render` writes
 * it from the trial, and each candidate's statistic by condition, as
 * `mise measure` writes it from a run.
 */
export interface Trial {
  given    : Readonly<Record<string, number>>
  measured : Readonly<Record<string, Readonly<Record<string, { error: number, mean: number }>>>>
}

/**
 * Reads the value `field` takes under each condition that sets it, out of the
 * changes each condition makes to the plan.
 */
export const setting = (
  changes : Readonly<Record<string, Readonly<Record<string, number>>>>,
  field   : string
): Record<string, number> =>
  Object.fromEntries(
    Object.entries(changes).flatMap(([condition, set]) =>
      field in set ? [[condition, set[field]!]] : []
    )
  )

/**
 * The coupling trial by launch spacing and the heading trial by turn, read
 * from the run `mise measure` last wrote out.
 */
export const TRIALS: Readonly<Record<'coupling' | 'heading', Trial>> = {
  coupling : { given: setting(conditions.coupling, 'spacing_m'), measured: trials.coupling },
  heading  : { given: setting(conditions.heading, 'turn_deg'), measured: trials.heading }
}

/**
 * Reads `rule` at `spacing` along the straight line between the two measured
 * spacings around it, the measured point itself where it sits on one.
 */
export function between(rule: Rule, spacing: number): Point {
  const { points } = rule
  const found      = points.findIndex(point => point.given >= spacing)
  const upper      = found < 0 ? points.length - 1 : Math.max(1, found)
  const [a, b]     = [points[upper - 1]!, points[upper]!]
  const share      = Math.min(1, Math.max(0, (spacing - a.given) / (b.given - a.given)))
  return {
    error : a.error + (b.error - a.error) * share,
    given : spacing,
    mean  : a.mean + (b.mean - a.mean) * share
  }
}

/**
 * Writes a measured point as its mean beside its standard error.
 */
export const reading = (point: Point): string => `${point.mean.toFixed(2)} ± ${point.error.toFixed(2)}`

/**
 * Lists each rule `names` holds out of `trial`, in that order, each colored by
 * its place in the chart series.
 */
export const rulesOf = (names: Names, trial: Trial): Rule[] =>
  Object.entries(names).map(([name, [label, short]], place) => ({
    color  : series(place),
    label,
    points : Object.entries(trial.measured[name]!)
      .map(([condition, { error, mean }]) => ({ error, given: trial.given[condition]!, mean }))
      .toSorted((a, b) => a.given - b.given),
    short
  }))
