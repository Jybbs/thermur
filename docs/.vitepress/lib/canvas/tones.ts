import * as common from '../common.ts'
import { type Identity, type Scheme, heatRamp, seriesOf, shadeOf } from '../identity.ts'
import source      from '../rendered/identity.json' with { type: 'json' }

/**
 * The front page's sky in one scheme, each color as a hex or a ready fill
 * string: its colorway beside the eight bands easing down from the night to the
 * horizon, the ridge softened and cleared for its fade, and the dusk bands of
 * the roost at the page's foot with the night cleared above them.
 */
interface SkyTones extends Omit<Way, 'ease' | 'low'> {
  bands      : readonly string[]
  ridgeClear : string
  ridgeSoft  : string
  roost      : { bands: readonly string[], clear: string }
}

/**
 * A colorway of the sky before its fills are written: the night at the top, the
 * color at the horizon, the ridge, how sharply the bands ease toward the
 * horizon, and the silhouette's color and opacities and the ridge's solid and
 * fading heights.
 */
interface Way {
  ease       : number
  fade       : number
  low        : string
  ridge      : string
  shade      : string
  shadeAlpha : readonly [number, number]
  solid      : number
  top        : string
}

/**
 * The number of bands the sky eases through from the night to the horizon.
 */
const BANDS = 8
/**
 * The shadow black the dark sky deepens its ground toward for the ridge and the
 * flock's silhouette, which the palette names no color for.
 */
const BLACK = '#000000'

/**
 * Writes `hex` at `alpha` as a CSS color, the hex itself where the color is
 * opaque.
 */
export const translucent = (hex: string, alpha: number): string =>
  alpha >= 1 ? hex : `rgb(${common.channels(hex).join(' ')} / ${alpha})`

/**
 * The colors a drawing draws in, read from the identity and written into fill
 * strings once per scheme, so a frame builds no string of its own.
 */
export class Tones {
  readonly heat      : readonly string[]
  readonly #flocks   : Record<Scheme, Map<number, readonly string[]>> = { dark: new Map(), light: new Map() }
  readonly #identity : Identity
  readonly #inks     : Record<Scheme, Map<number, string>> = { dark: new Map(), light: new Map() }
  readonly #skies    = new Map<Scheme, SkyTones>()

  constructor(identity: Identity = source) {
    this.#identity = identity
    this.heat      = heatRamp(identity)
  }

  /**
   * Reads the colorway of the sky in `scheme`: the settled dusk in the dark
   * scheme, and the dawn in the light one, its cream warming to a peach at the
   * horizon over a deep warm ridge.
   */
  #way(scheme: Scheme): Way {
    const ground = shadeOf('ground', scheme, this.#identity)
    if (scheme === 'dark') {
      const lit = common.mix(ground, seriesOf(0, 'dark', this.#identity), 0.12)
      const low = common.mix(lit, shadeOf('ground', 'light', this.#identity), 0.035)
      return {
        ease       : 1.7,
        fade       : 260,
        low        : low,
        ridge      : common.mix(ground, BLACK, 0.32),
        shade      : common.mix(ground, BLACK, 0.48),
        shadeAlpha : [0.55, 0.95],
        solid      : 0,
        top        : ground
      }
    }
    return {
      ease       : 1.5,
      fade       : 48,
      low        : common.mix(ground, this.heat[0]!, 0.42),
      ridge      : common.mix(this.ink('light'), this.heat[5]!, 0.22),
      shade      : this.ink('light'),
      shadeAlpha : [0.42, 0.85],
      solid      : 22,
      top        : ground
    }
  }

  /**
   * Reads the color of the flock's gradient `t` of the way from its foot in the
   * scheme's ink to its head in the fire's series color, at `alpha`. The
   * gradient is written in 33 steps once per scheme and opacity, and each
   * caller passes one of a few fixed opacities, so the table stays small.
   */
  flock(scheme: Scheme, t: number, alpha: number): string {
    const steps = common.cached(this.#flocks[scheme], alpha, () => {
      const [foot, head] = [this.ink(scheme), seriesOf(0, scheme, this.#identity)]
      return Array.from({ length: 33 }, (_, k) =>
        translucent(common.mix(foot, head, common.smoothstep(0, 0.8, k / 32)), alpha))
    })
    return steps[Math.round(common.clamp(t) * 32)]!
  }

  /**
   * Reads the ink of `scheme`, at `alpha` where one is given.
   */
  ink(scheme: Scheme, alpha = 1): string {
    return common.cached(this.#inks[scheme], alpha, () =>
      translucent(shadeOf('ink', scheme, this.#identity), alpha))
  }

  /**
   * Reads the sky in `scheme`, writing its fills on the first read.
   */
  sky(scheme: Scheme): SkyTones {
    return common.cached(this.#skies, scheme, () => {
      const { ease, low, ...kept } = this.#way(scheme)
      const tone                   = (u: number): string => common.mix(kept.top, low, u ** ease)
      return {
        ...kept,
        bands      : Array.from({ length: BANDS }, (_, k) => tone(k / (BANDS - 1))),
        ridgeClear : translucent(kept.ridge, 0),
        ridgeSoft  : translucent(kept.ridge, 0.85),

        roost: {
          bands : [1, 2, 3, 4].map(k => common.mix(kept.top, low, (k / 4) ** ease * 0.75)),
          clear : translucent(kept.top, 0)
        }
      }
    })
  }
}
