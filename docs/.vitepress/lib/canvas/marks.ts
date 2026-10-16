import { cached }                  from '../common.ts'
import type { Identity, Scheme }   from '../identity.ts'
import { glyphsOf }                from '../scene.ts'
import { type Tones, translucent } from './tones.ts'

/**
 * A short fire as `Marks.fire` draws it: on the ground at `x` and `y`, its
 * flame `half` pixels either side of its center under a glow reaching `glow`
 * pixels, 120 unless given, times `0.6 + 0.6 × fire`, where `fire` is the
 * fire's strength, which also sets how opaque both are.
 */
interface Flame {
  fire   : number
  half   : number
  scheme : Scheme
  x      : number
  y      : number
  glow?  : number
}

/**
 * The text of each SVG the public folder holds, by its path from here.
 */
const publicSvgs = import.meta.glob<string>('../../../public/*.svg', {
  eager  : true,
  import : 'default',
  query  : '?raw'
})

/**
 * The pieces a drawing draws over and over, each built once: the traced birds
 * as paths, and the fire's glow as a small canvas per scheme, placed scaled and
 * faded every frame rather than drawn through a gradient.
 */
export class Marks {
  readonly count    : number
  readonly #glows   = new Map<Scheme, HTMLCanvasElement>()
  readonly #glyphs  : readonly Path2D[]
  readonly #tones   : Tones

  constructor(glyphs: readonly string[], tones: Tones) {
    this.count   = glyphs.length
    this.#glyphs = glyphs.map(d => new Path2D(d))
    this.#tones  = tones
  }

  /**
   * Builds the marks a drawing in `tones` draws, the traced birds read from the
   * dark scheme's mark the public folder holds.
   */
  static traced(tones: Tones, identity: Identity): Marks {
    return new Marks(glyphsOf(publicSvgs[`../../../public/${identity.marks.mark.dark}`]!), tones)
  }

  /**
   * Draws the fire's glow in `scheme` once into a canvas 256 pixels across,
   * fading from its center out to nothing at its edge.
   */
  #glow(scheme: Scheme): HTMLCanvasElement {
    return cached(this.#glows, scheme, () => {
      const [ember, kindle] = [this.#tones.heat[1]!, this.#tones.heat[0]!]
      const strong          = scheme === 'dark' ? [0.2, 0.07] : [0.14, 0.05]
      const glow            = document.createElement('canvas')
      glow.width            = 256
      glow.height           = 256
      const context         = glow.getContext('2d')!
      const gradient        = context.createRadialGradient(128, 128, 0, 128, 128, 128)
      gradient.addColorStop(0, translucent(ember, strong[0]!))
      gradient.addColorStop(0.5, translucent(kindle, strong[1]!))
      gradient.addColorStop(1, translucent(kindle, 0))
      context.fillStyle = gradient
      context.fillRect(0, 0, 256, 256)
      return glow
    })
  }

  /**
   * Draws traced bird `glyph` with its head at `x` and `y`, turned to `angle`
   * and `length` pixels long, in `color`, leaving the transform set.
   */
  bird(
    ctx    : CanvasRenderingContext2D,
    dpr    : number,
    glyph  : number,
    x      : number,
    y      : number,
    angle  : number,
    length : number,
    color  : string
  ): void {
    const c = Math.cos(angle) * length * dpr
    const s = Math.sin(angle) * length * dpr
    ctx.setTransform(c, s, -s, c, x * dpr, y * dpr)
    ctx.fillStyle = color
    ctx.fill(this.#glyphs[glyph]!)
  }

  /**
   * Draws a short fire, a hairline of flame on the ground under a soft glow,
   * and nothing while the fire is out.
   */
  fire(ctx: CanvasRenderingContext2D, { fire, glow = 120, half, scheme, x, y }: Flame): void {
    if (fire <= 0) return
    const radius = glow * (0.6 + 0.6 * fire)
    ctx.save()
    ctx.globalAlpha = fire
    ctx.drawImage(this.#glow(scheme), x - radius, y - radius, radius * 2, radius * 2)
    ctx.lineCap     = 'round'
    ctx.globalAlpha = 0.35 * fire
    ctx.strokeStyle = this.#tones.heat[1]!
    ctx.lineWidth   = 5
    ctx.beginPath()
    ctx.moveTo(x - half, y)
    ctx.quadraticCurveTo(x, y - 5, x + half, y)
    ctx.stroke()
    ctx.globalAlpha = 0.55 + 0.4 * fire
    ctx.strokeStyle = this.#tones.heat[0]!
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(x - half * 0.9, y)
    ctx.quadraticCurveTo(x, y - 5, x + half * 0.9, y)
    ctx.stroke()
    ctx.restore()
  }
}
