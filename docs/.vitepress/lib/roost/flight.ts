import * as common   from '../common.ts'
import { Composite } from '../front/composite.ts'
import { FLIGHT }    from '../front/flock.ts'
import { Fire }      from './fire.ts'

/**
 * The flight the view plays: the drones around the fire, how long they fly,
 * the frames kept a second, the simulated seconds shown in one, the radius the
 * drawing places the roost's disk at, in meters, as the front page places its
 * own, and the moment shown while motion stands still, in seconds. The drone
 * the camera follows is the one of the 200 that keeps the most others within
 * 150 m of it at the fewest it ever keeps over the flight, 73.
 */
export const PLAN = { drones: 200, hz: 5, led: 50, pace: 12, radius: 500, seconds: 300, seed: FLIGHT.seed, still: 160 } as const
/**
 * The frames one flight keeps.
 */
export const FRAMES = PLAN.seconds * PLAN.hz + 1
/**
 * The seconds over which the point a camera follows settles onto its drone.
 */
const FOLLOW = 3
/**
 * One flight of the site's parity-tested copy of the package's composite under
 * the package's own settings, the plan's drones over the roost's disk placed at
 * the plan's radius, pushed by the heat of a fire lit at the disk's middle.
 * Each step is flown the first time a frame past it is read and its frame
 * kept, so the flight plays again without flying, in arrays built once at the
 * flight's length. Beside each drone's place, each frame keeps the
 * polarization, the mean distance to the nearest neighbor, the mean, lowest,
 * and highest height, and the place a camera following a drone moves over,
 * every length in meters.
 */
export class Flight {
  readonly count        = PLAN.drones
  readonly fire         = new Fire()
  readonly followed     = new Float32Array(FRAMES * 2)
  readonly height       = new Float32Array(FRAMES)
  readonly high         = new Float32Array(FRAMES)
  readonly led          = PLAN.led
  readonly low          = new Float32Array(FRAMES)
  readonly places       : Float32Array
  readonly polarization = new Float32Array(FRAMES)
  readonly roost        : Composite['roost']
  readonly spacing      = new Float32Array(FRAMES)
  readonly #composite   : Composite
  readonly #draws       : Float64Array
  #flown                = 0
  readonly #increments  : Float64Array
  readonly #normal      : () => number
  readonly #pushes      : Float64Array
  readonly #random      = common.mulberry(PLAN.seed)

  constructor() {
    const drones     = this.count
    this.places      = new Float32Array(FRAMES * drones * 3)
    this.#composite  = new Composite(drones)
    this.#draws      = new Float64Array(drones)
    this.#increments = new Float64Array(drones * 3)
    this.#pushes     = new Float64Array(drones * 3)
    this.roost       = this.#composite.roost
    this.roost.radius_m = PLAN.radius
    this.#normal     = common.gaussian(this.#random)
    this.#composite.launch(this.#normal)
    this.#keep()
  }

  #keep(): void {
    const { positions, velocities } = this.#composite
    const frame = this.#flown
    const count = this.count
    let gap = 0, high = -Infinity, hx = 0, hy = 0, hz = 0, low = Infinity, z = 0
    for (let i = 0; i < count; i++) {
      const at    = i * 3
      const speed = Math.hypot(velocities[at]!, velocities[at + 1]!, velocities[at + 2]!)
      hx  += velocities[at]! / speed
      hy  += velocities[at + 1]! / speed
      hz  += velocities[at + 2]! / speed
      z   += positions[at + 2]!
      low  = Math.min(low, positions[at + 2]!)
      high = Math.max(high, positions[at + 2]!)
      let best = Infinity
      for (let other = 0; other < positions.length; other += 3) {
        if (other === at) continue
        const dx = positions[at]! - positions[other]!
        const dy = positions[at + 1]! - positions[other + 1]!
        const dz = positions[at + 2]! - positions[other + 2]!
        best     = Math.min(best, dx * dx + dy * dy + dz * dz)
      }
      gap += Math.sqrt(best)
    }
    const ease                    = frame === 0 ? 1 : 1 - Math.exp(-1 / (FOLLOW * PLAN.hz))
    for (let k = 0; k < 2; k++) {
      const last                  = frame === 0 ? 0 : this.followed[frame * 2 - 2 + k]!
      this.followed[frame * 2 + k] = last + (positions[this.led * 3 + k]! - last) * ease
    }
    this.polarization[frame]      = Math.hypot(hx, hy, hz) / count
    this.spacing[frame]           = gap / count
    this.height[frame]            = z / count
    this.low[frame]               = low
    this.high[frame]              = high
    this.places.set(positions, frame * positions.length)
    this.#flown++
  }

  /**
   * Flies on until `frame` is kept.
   */
  fly(frame: number): void {
    const step = this.#composite.model.plan.step_s
    const root = Math.sqrt(step)
    const per  = Math.round(1 / (PLAN.hz * step))
    const [east, north] = this.roost.center_m
    while (this.#flown <= Math.min(frame, FRAMES - 1)) {
      for (let k = 0; k < per; k++) {
        const seconds = ((this.#flown - 1) * per + k) * step
        this.fire.push(seconds, east, north, this.#composite.positions, this.#pushes)
        for (let i = 0; i < this.#increments.length; i++) this.#increments[i] = this.#normal() * root
        for (let i = 0; i < this.#draws.length; i++) this.#draws[i] = this.#random()
        this.#composite.hear(this.#random)
        this.#composite.advance(this.#increments, this.#draws, this.#pushes)
      }
      this.#keep()
    }
  }

  /**
   * Flies the flight to the still moment a slice at a time, so a page that
   * stands still stays responsive while it is flown.
   */
  async ready(): Promise<void> {
    const last = PLAN.still * PLAN.hz + 1
    if (this.#flown > last) return
    const from = performance.now()
    while (this.#flown <= last && performance.now() - from < 24) this.fly(this.#flown)
    await new Promise<void>(done => { setTimeout(done) })
    await this.ready()
  }
}

let flown: Flight | undefined

/**
 * Reads the one flight the page flies, flown once however many views play it.
 */
export const flight = (): Flight => flown ??= new Flight()
