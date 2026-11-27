import { BANDS }             from './burn.ts'
import { type Flight, PLAN } from './flight.ts'
import type { Player }       from './player.ts'
import type { Camera, Shot } from './scene.ts'

/**
 * The meters to either side of the followed drone the view holds.
 */
const HALF = 180

/**
 * A drone's length over the flight, in meters, held between the fewest and
 * most pixels a bird takes.
 */
const BIRD = [11, 4, 18] as const

/**
 * Aims `camera` over the ground with the followed drone, 38 degrees up,
 * facing west.
 */
const follow = (camera: Camera, player: Player, w: number, height: number): void => {
  const { followed } = player.flight
  camera.x      = player.read(followed, 2, 0)
  camera.y      = player.read(followed, 2, 1)
  camera.facing = -Math.PI / 2
  camera.pitch  = 38 * Math.PI / 180
  camera.scale  = Math.min(w / 2, height * 0.9) / HALF
  camera.away   = 4 * HALF
  camera.across = w / 2
  camera.down   = height * 0.48 + 100 * Math.cos(camera.pitch) * camera.scale
}

/**
 * Aims `camera` over the whole roost from 34 degrees up, facing west.
 */
const whole = (camera: Camera, player: Player, w: number, height: number): void => {
  const { center_m: [east, north], radius_m } = player.flight.roost
  camera.x      = east
  camera.y      = north
  camera.facing = -Math.PI / 2
  camera.pitch  = 34 * Math.PI / 180
  camera.away   = 3.4 * radius_m
  camera.scale  = Math.min(w * 0.46, height * 0.655) / radius_m
  camera.across = w / 2
  camera.down   = height * 0.5
}

const near : Camera = { across: 0, away: 0, down: 0, facing: 0, pitch: 0, scale: 1, x: 0, y: 0 }
const far  : Camera = { ...near }

/**
 * The view moving over the ground with the ringed drone, lifted toward the
 * whole roost by as much as the player's `lift` reads, over the fire drawn in
 * bands.
 */
export const SHOT: Shot = {
  aim   : (camera, player, w, height) => {
    follow(near, player, w, height)
    if (player.lift === 0) return void Object.assign(camera, near)
    whole(far, player, w, height)
    for (const key of ['across', 'away', 'down', 'facing', 'pitch', 'x', 'y'] as const) camera[key] = near[key] + (far[key] - near[key]) * player.lift
    camera.scale = near.scale * (far.scale / near.scale) ** player.lift
  },
  bird  : BIRD,
  burn  : BANDS,
  drops : true,
  grid  : 50,
  ring  : true,
  trail : 16
}

/**
 * Writes the sentence under the charts for `flight`: its drones, their roost,
 * the fire spreading from its middle at the front's speed, the pace, and what
 * the bands say.
 */
export const noteOf = ({ count, fire }: Flight): string =>
  `${count} drones over a roost ${PLAN.radius * 2 / 1000} km across, a fire spreading from its middle at ${fire.front.at(1)} m/s, `
  + `the package's default threat, the drones turning from its heat. The view follows the ringed drone, ${PLAN.seconds} s shown ${PLAN.pace} times as fast. ${BANDS.said}`
