import source                   from '../rendered/composite.json' with { type: 'json' }
import type { Playing, Watched } from './flight.ts'
import { Flying }                from './flying.ts'
import { Replayed }              from './replayed.ts'

/**
 * The flock sizes the Lookout's slider sets: from a few tens of drones to
 * 3,000, the largest flock the radio's table was checked at against an exact
 * search of every pair, by tens, the plan's default first, and the stops an
 * arrow key steps between.
 */
export const DRONES = {
  max   : 3000,
  min   : 20,
  step  : 10,
  stops : [20, 50, 100, 200, 500, 1000, 2000, 3000],
  value : source.plan.drones
} as const

/**
 * Builds the flight of `count` drones the Lookout plays, flown in the page from
 * the plan's seed where `served` is `null`, and otherwise the flight `thermur
 * watch` flies through the package's graph at the address `served`.
 */
export async function played(count: number, served: string | null): Promise<Playing> {
  if (served === null) return new Flying(count, source.plan.seed)
  const answer = await fetch(`${served}?${new URLSearchParams({ drones: String(count) })}`)
  return new Replayed(await answer.json() as Watched)
}

/**
 * Reads the flock's size aloud, naming the plan's default where it stands
 * there.
 */
export const spoken = (count: number): string =>
  `${count.toLocaleString('en-US')} drones${count === DRONES.value ? ', the default' : ''}`
