import source from '../rendered/threat.json' with { type: 'json' }

/**
 * The default parametric threat and the constants its air is worked out from,
 * as `mise render` writes them from the package's record.
 */
export type Threat = typeof source

/**
 * A small model of the package's parametric threat, working out from one record
 * what its dataset holds at any moment rather than at each snapshot alone, so a
 * drawing can run the front between the snapshots the dataset writes.
 */
export class Front {
  readonly interval  : number
  readonly lowest    : number
  readonly snapshots : number
  readonly spacing   : number
  readonly span      : number
  readonly #record   : Threat

  constructor(record: Threat = source) {
    const { interval_s, level_m, snapshots, spacing_m } = record.grid
    this.interval  = interval_s
    this.lowest    = level_m / 2
    this.snapshots = snapshots
    this.spacing   = spacing_m
    this.span      = interval_s * (snapshots - 1)
    this.#record   = record
  }

  /**
   * Reads the distance of the center of the last column from the western edge,
   * in meters.
   */
  get far(): number {
    return this.center(this.#record.grid.columns - 1)
  }

  /**
   * Reads the air temperature `x` meters east of the western edge and `height`
   * meters above the ground after `seconds`, in kelvin: the base state warmed
   * by the anomaly the profile carries, and cooled with height by the Exner
   * function.
   */
  air(height: number, seconds: number, x: number): number {
    const { anomaly_k, base_k, gravity_m_s2, heat_capacity_j_kg_k } = this.#record
    const exner = 1 - gravity_m_s2 * height / (heat_capacity_j_kg_k * base_k)
    return (base_k + anomaly_k * this.profile(seconds, x)) * exner
  }

  /**
   * Reads the distance the front has advanced from the western edge after
   * `seconds`, in meters.
   */
  at(seconds: number): number {
    return this.#record.speed_m_s * seconds
  }

  /**
   * Reads the distance of the center of `column` from the western edge, in
   * meters.
   */
  center(column: number): number {
    return (column + 0.5) * this.spacing
  }

  /**
   * Reads the share of the anomaly that reaches `x` meters east of the western
   * edge after `seconds`, one well behind the front and zero well ahead of it.
   */
  profile(seconds: number, x: number): number {
    return (1 - Math.tanh((x - this.at(seconds)) / this.#record.edge_m)) / 2
  }
}
