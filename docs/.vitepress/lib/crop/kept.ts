import source     from '../rendered/crop.json' with { type: 'json' }
import { Glide } from './glide.ts'

/**
 * The crop's heights and the sample's shape, which every menu and the strip
 * read their figures from.
 */
export type CropModel = typeof source

/**
 * The height the menus keep at most, in meters.
 */
export const MOST_M = 1000

/**
 * What the menus keep, each part gliding: the share of the domain's length from
 * its western end to `west` and to `east`, the height `top` reaches in meters,
 * and the `first` snapshot, with `reach` the height the record's defaults
 * reach.
 */
export class Kept {
  readonly east  = new Glide(1)
  readonly first = new Glide(0)
  readonly model : CropModel
  readonly reach : number
  readonly top   : Glide
  readonly west  = new Glide(0)
  readonly #all  : readonly Glide[]

  constructor(model: CropModel = source) {
    this.model = model
    this.reach = model.floor_m + model.spacing_m * (model.levels - 1)
    this.top   = new Glide(this.reach)
    this.#all  = [this.east, this.first, this.top, this.west]
  }

  /**
   * Reads the share of the most the menus can keep that they keep now, by
   * length, height, and snapshots.
   */
  get share(): number {
    const { east, first, model, top, west } = this
    const snapshots                         = model.times_s.length
    return (east.value - west.value) * (top.value / MOST_M) * ((snapshots - first.value) / snapshots)
  }

  /**
   * Puts every part at its target at rest.
   */
  settle(): void {
    for (const glide of this.#all) glide.settle()
  }

  /**
   * Moves every part on by `dt` seconds.
   */
  step(dt: number): void {
    for (const glide of this.#all) glide.step(dt)
  }
}
