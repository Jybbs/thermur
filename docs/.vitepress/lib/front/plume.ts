import type { Heat } from './heat.ts'

/**
 * The shape of a plume of hot air: how far it leans downwind per pixel it
 * rises, how fast it widens, the height its heat falls off over, its width at
 * the ground, and how strong it is at its core.
 */
interface Rise {
  core   : number
  gain   : number
  lean   : number
  reach  : number
  spread : number
}

/**
 * A fire on the ground at `x` and `y` and the plume of hot air it sends up,
 * leaning with the wind, widening as it rises, and swaying with `t`, its
 * strength the fire's.
 */
export class Plume implements Heat {
  fire   = 0
  t      = 0
  x      = 0
  y      = 0
  readonly rise : Rise

  constructor(rise: Rise) {
    this.rise = rise
  }

  heat(px: number, py: number): number {
    if (this.fire <= 0 || py > this.y + 10) return 0
    const { core, gain, lean, reach, spread } = this.rise
    const rise                                = Math.max(0, this.y - py)
    const center                              = this.x + lean * rise + 10 * Math.sin(rise / 70 + this.t * 0.5)
    const width                               = core * (0.6 + 0.6 * this.fire) + spread * rise
    return gain * this.fire * Math.exp(-((px - center) ** 2) / (2 * width * width))
         * Math.exp(-rise / reach) * Math.sqrt(core / width)
  }
}
