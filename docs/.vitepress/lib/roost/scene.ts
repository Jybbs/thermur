import type { Marks }          from '../canvas/marks.ts'
import type { Geometry, Part } from '../canvas/part.ts'
import type { Tones }          from '../canvas/tones.ts'
import * as common             from '../common.ts'
import { shadeOf }             from '../identity.ts'
import { FRAMES, PLAN }        from './flight.ts'
import type { Player }         from './player.ts'

/**
 * A camera over the roost: the point on the ground it looks at, in meters, the
 * way it faces around the compass and down, in radians, how far off it stands,
 * in meters, infinite for a view with no perspective, and where on the canvas
 * the point falls and how many pixels a meter spans there.
 */
export interface Camera {
  across  : number
  away    : number
  down    : number
  facing  : number
  pitch   : number
  scale   : number
  x       : number
  y       : number
}

/**
 * How a scene draws the fire of a burning flight: the fire on the ground under
 * the flock, drawn through the scene's projection, and the color of the drone
 * at `index` at the moment shown, the flock's own where the heat reaches it
 * too faintly to show, beside the sentence the view's note closes on, which
 * says in words what the colors mean.
 */
export interface Burn {
  ground : (ctx: CanvasRenderingContext2D, g: Geometry, scene: Scene, player: Player) => void
  said   : string
  tone   : (player: Player, index: number, scheme: Geometry['scheme']) => string | undefined
}

/**
 * How a scene places its camera each frame on a canvas `w` by `h`, and what it
 * draws beside the flock.
 */
export interface Shot {
  burn?   : Burn
  aim     : (camera: Camera, player: Player, w: number, h: number) => void
  /**
   * A drone's length, in meters of the ground it is drawn over, held between
   * the fewest and most pixels a bird takes.
   */
  bird    : readonly [number, number, number]
  drops   : boolean
  grid    : number
  trail   : number
  ring?   : boolean
}

/**
 * The points a circle on the ground is traced through.
 */
const ROUND = 120
/**
 * The frames between two points of a trail.
 */
const STRIDE = 3
/**
 * The opacities a trail fades through from its tail to its drone, one path
 * stroked for each.
 */
const FADES = [0.04, 0.08, 0.13, 0.19, 0.26, 0.34] as const

/**
 * The flock over the roost's disk, through the camera `shot` aims: the disk
 * and a grid on the ground inside it, a fading trail behind each drone, and each drone one of the traced
 * birds turned the way it flies.
 */
export class Scene implements Part {
  readonly #camera : Camera = { across: 0, away: Infinity, down: 0, facing: 0, pitch: Math.PI / 2, scale: 1, x: 0, y: 0 }
  readonly #glyphs : Uint8Array
  readonly #marks  : Marks
  readonly #player : Player
  readonly #shot   : Shot
  readonly #shown  : () => void
  readonly #tones  : Tones
  readonly #at     = new Float32Array(3)
  readonly #from   = new Float32Array(3)
  readonly #out    = new Float32Array(3)
  readonly #led    = new Float32Array(2)

  constructor(marks: Marks, tones: Tones, player: Player, shot: Shot, shown: () => void) {
    const random = common.mulberry(PLAN.seed)
    this.#marks  = marks
    this.#tones  = tones
    this.#player = player
    this.#shot   = shot
    this.#shown  = shown
    this.#glyphs = Uint8Array.from({ length: player.flight.count }, () => Math.floor(random() * marks.count))
  }

  /**
   * Projects a point onto the canvas into `#out`: its place across and down,
   * and its scale in pixels a meter.
   */
  #project(x: number, y: number, z: number): void {
    const c     = this.#camera
    const dx    = x - c.x, dy = y - c.y
    const right = dx * Math.cos(c.facing) - dy * Math.sin(c.facing)
    const ahead = dx * Math.sin(c.facing) + dy * Math.cos(c.facing)
    const depth = ahead * Math.cos(c.pitch) - z * Math.sin(c.pitch)
    const up    = ahead * Math.sin(c.pitch) + z * Math.cos(c.pitch)
    const near  = c.away === Infinity ? 1 : c.away / Math.max(c.away + depth, c.away * 0.1)
    this.#out[0] = c.across + right * near * c.scale
    this.#out[1] = c.down - up * near * c.scale
    this.#out[2] = near * c.scale
  }

  /**
   * Traces the circle `radius` meters around the roost's middle, `z` meters
   * above the ground, as a path of `ctx`, which the caller fills or strokes.
   */
  circle(ctx: CanvasRenderingContext2D, radius: number, z = 0): void {
    const [east, north] = this.#player.flight.roost.center_m
    ctx.beginPath()
    for (let k = 0; k <= ROUND; k++) {
      const turn = k / ROUND * Math.PI * 2
      this.#project(east + radius * Math.cos(turn), north + radius * Math.sin(turn), z)
      if (k === 0) ctx.moveTo(this.#out[0]!, this.#out[1]!)
      else ctx.lineTo(this.#out[0]!, this.#out[1]!)
    }
  }

  #line(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number): void {
    this.#project(x0, y0, 0)
    ctx.moveTo(this.#out[0]!, this.#out[1]!)
    this.#project(x1, y1, 0)
    ctx.lineTo(this.#out[0]!, this.#out[1]!)
  }

  #ground(ctx: CanvasRenderingContext2D, g: Geometry): void {
    const { center_m: [east, north], radius_m } = this.#player.flight.roost
    const ink = this.#tones.ink(g.scheme, 1)
    this.circle(ctx, radius_m)
    ctx.fillStyle   = common.mix(shadeOf('ground', g.scheme), g.scheme === 'dark' ? '#f2f1ee' : '#17171b', 0.045)
    ctx.fill()
    ctx.strokeStyle = ink
    ctx.globalAlpha = 0.5
    ctx.lineWidth   = 1.25
    ctx.stroke()
    ctx.globalAlpha = 0.12
    ctx.lineWidth   = 1
    const grid = this.#shot.grid
    ctx.beginPath()
    for (let offset = -radius_m + grid; offset < radius_m; offset += grid) {
      const half = Math.sqrt(radius_m ** 2 - offset ** 2)
      this.#line(ctx, east + offset, north - half, east + offset, north + half)
      this.#line(ctx, east - half, north + offset, east + half, north + offset)
    }
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  #trails(ctx: CanvasRenderingContext2D, g: Geometry): void {
    const { count, places } = this.#player.flight
    const frame      = this.#player.frame
    const points     = Math.floor(this.#shot.trail * PLAN.hz / STRIDE)
    const part       = Math.ceil(points / FADES.length)
    const last       = frame - frame % STRIDE
    ctx.strokeStyle  = this.#tones.ink(g.scheme, 1)
    ctx.lineWidth    = 1
    ctx.lineJoin     = 'round'
    for (let fade = 0; fade < FADES.length; fade++) {
      ctx.globalAlpha = FADES[fade]!
      ctx.beginPath()
      for (let index = 0; index < count; index++) {
        let drawn = false
        for (let k = 0; k <= part; k++) {
          const at = last - (points - fade * part - k) * STRIDE
          if (at < 0 || at > frame) continue
          const p = (at * count + index) * 3
          this.#project(places[p]!, places[p + 1]!, places[p + 2]!)
          if (drawn) ctx.lineTo(this.#out[0]!, this.#out[1]!)
          else ctx.moveTo(this.#out[0]!, this.#out[1]!)
          drawn = true
        }
        if (fade < FADES.length - 1) continue
        this.#player.place(index, this.#at)
        this.#project(this.#at[0]!, this.#at[1]!, this.#at[2]!)
        ctx.lineTo(this.#out[0]!, this.#out[1]!)
      }
      ctx.stroke()
    }
    ctx.globalAlpha = 1
  }

  #drops(ctx: CanvasRenderingContext2D, g: Geometry): void {
    ctx.fillStyle   = this.#tones.ink(g.scheme, 1)
    ctx.globalAlpha = 0.28
    for (let index = 0; index < this.#player.flight.count; index++) {
      this.#player.place(index, this.#at)
      this.#project(this.#at[0]!, this.#at[1]!, 0)
      ctx.fillRect(this.#out[0]! - 0.75, this.#out[1]! - 0.75, 1.5, 1.5)
    }
    ctx.globalAlpha = 1
  }

  #birds(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number): void {
    const { count, led, places } = this.#player.flight
    const frame      = Math.min(this.#player.frame, FRAMES - 2)
    const color      = this.#tones.flock(g.scheme, 1, 1)
    const burn       = this.#shot.burn
    for (let index = 0; index < count; index++) {
      const p = (frame * count + index) * 3
      const n = p + count * 3
      this.#project(places[p]!, places[p + 1]!, places[p + 2]!)
      this.#from.set(this.#out)
      this.#project(places[n]!, places[n + 1]!, places[n + 2]!)
      const angle  = Math.atan2(this.#out[1]! - this.#from[1]!, this.#out[0]! - this.#from[0]!)
      this.#player.place(index, this.#at)
      this.#project(this.#at[0]!, this.#at[1]!, this.#at[2]!)
      const [meters, least, most] = this.#shot.bird
      const length = common.clamp(meters * this.#out[2]!, least, most)
      if (index === led) {
        this.#led[0] = this.#out[0]! - Math.cos(angle) * length / 2
        this.#led[1] = this.#out[1]! - Math.sin(angle) * length / 2
      }
      this.#marks.bird(ctx, dpr, this.#glyphs[index]!, this.#out[0]!, this.#out[1]!, angle, length, burn?.tone(this.#player, index, g.scheme) ?? color)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  /**
   * Rings the followed drone.
   */
  #ring(ctx: CanvasRenderingContext2D, g: Geometry): void {
    ctx.strokeStyle = this.#tones.ink(g.scheme, 1)
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.arc(this.#led[0]!, this.#led[1]!, 16, 0, Math.PI * 2)
    ctx.stroke()
  }

  paint(ctx: CanvasRenderingContext2D, g: Geometry, dpr: number): void {
    this.#shot.aim(this.#camera, this.#player, g.w, g.h)
    this.#ground(ctx, g)
    this.#shot.burn?.ground(ctx, g, this, this.#player)
    if (this.#shot.drops) this.#drops(ctx, g)
    this.#trails(ctx, g)
    this.#birds(ctx, g, dpr)
    if (this.#shot.ring) this.#ring(ctx, g)
    this.#shown()
  }

  step(dt: number): void {
    this.#player.step(dt)
  }
}
