import * as common from '../common.ts'

/**
 * Where the links start, following the picked bird at `x` and `y` with the
 * velocity `vx` and `vy`, and how far the links have faded in, `w`, from 0 to
 * 1.
 */
interface Anchor {
  vx : number
  vy : number
  w  : number
  x  : number
  y  : number
}

/**
 * How many of the picked bird's nearest neighbors a linked bird may stand among
 * and keep its link, so a bird hovering at seventh place keeps its link rather
 * than trading it back and forth.
 */
const KEPT = 9
/**
 * How fast a link reaches out and draws back, in reaches a second, and how long
 * each of the seven waits behind the one nearer, in reaches.
 */
const REACH = { back: 1 / 0.18, out: 1 / 0.32, stagger: 0.22 } as const

/**
 * The hairlines from the picked bird to its seven nearest neighbors. Each of
 * the seven links holds a bird, and when its bird drops out, the link draws
 * back to the picked bird before it reaches out to the bird that took its
 * place, so no more than seven ever show and none swings across the flock.
 * Every array is made when the links are built, so a frame makes nothing new.
 */
export class Links {
  readonly anchor : Anchor = { vx: 0, vy: 0, w: 0, x: 0, y: 0 }
  readonly #best  = new Float32Array(KEPT)
  readonly #bird  = new Int32Array(common.NEIGHBORS).fill(-1)
  readonly #near  = new Int32Array(KEPT)
  readonly #next  = new Int32Array(common.NEIGHBORS).fill(-1)
  readonly #reach = new Float32Array(common.NEIGHBORS)

  /**
   * Gives each link the bird it reaches for: its own while that bird stays
   * among the nine nearest, and otherwise the nearest of the seven that no link
   * holds or waits for.
   */
  #assign(): void {
    for (let link = 0; link < common.NEIGHBORS; link++) {
      const bird = this.#bird[link]!
      const next = this.#next[link]!
      if (bird >= 0 && this.#rank(bird) < KEPT) {
        this.#next[link] = -1
        continue
      }
      if (next >= 0 && this.#rank(next) < common.NEIGHBORS) continue
      this.#next[link] = -1
      for (let k = 0; k < common.NEIGHBORS; k++) {
        if (this.#held(this.#near[k]!)) continue
        this.#next[link] = this.#near[k]!
        break
      }
    }
  }

  /**
   * Reports whether a link holds `bird` or waits to reach for it.
   */
  #held(bird: number): boolean {
    return this.#bird.includes(bird) || this.#next.includes(bird)
  }

  /**
   * Reads where `bird` stands among the nine nearest, or nine where it stands
   * outside them.
   */
  #rank(bird: number): number {
    const rank = this.#near.indexOf(bird)
    return rank === -1 ? KEPT : rank
  }

  /**
   * Fades the links in toward 1 where they reach out and out toward 0
   * otherwise, at once while the drawing stands still.
   */
  fade(dt: number, out: boolean, still: boolean): void {
    const target  = out ? 1 : 0
    this.anchor.w = still ? target : this.anchor.w + (target - this.anchor.w) * (1 - Math.exp(-dt * 7))
  }

  /**
   * Raises the warmth each linked bird should take in `into`, rising as its
   * link arrives.
   */
  glow(into: Float32Array): void {
    for (let link = 0; link < common.NEIGHBORS; link++) {
      const bird = this.#bird[link]!
      if (bird >= 0) into[bird] = Math.max(into[bird]!, 0.8 * common.smoothstep(0.3, 1, this.#reach[link]!))
    }
  }

  /**
   * Draws each link that has set out as a line from where the links start
   * toward its bird, as far as it has reached, in the stroke `ctx` holds.
   */
  paint(ctx: CanvasRenderingContext2D, xs: Float32Array, ys: Float32Array): void {
    const { x, y } = this.anchor
    for (let link = 0; link < common.NEIGHBORS; link++) {
      const bird  = this.#bird[link]!
      const eased = common.smoothstep(0, 1, this.#reach[link]!)
      if (bird < 0 || eased < 0.01) continue
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.lineTo(x + (xs[bird]! - x) * eased, y + (ys[bird]! - y) * eased)
      ctx.stroke()
    }
  }

  /**
   * Finds the picked bird's nearest neighbors, gives each link its bird, and
   * moves where the links start toward the picked bird on a critically damped
   * spring, stepped implicitly so a long frame never overshoots. The links
   * start at the picked bird itself while they are hidden and while the drawing
   * stands still.
   *
   * Args:
   *   index : The picked bird.
   *   xs    : Each bird's place across the drawing.
   *   ys    : Each bird's place down the drawing.
   *   dt    : The seconds since the last frame.
   *   still : Whether the drawing stands still.
   */
  pick(index: number, xs: Float32Array, ys: Float32Array, dt: number, still: boolean): void {
    common.nearest(this.#best, index, this.#near, xs, ys)
    this.#assign()
    const anchor = this.anchor
    if (anchor.w < 0.02 || still) {
      anchor.x  = xs[index]!
      anchor.y  = ys[index]!
      anchor.vx = 0
      anchor.vy = 0
    }
    const damping = 1 / (1 + 32 * dt + 256 * dt * dt)
    anchor.vx     = (anchor.vx + (xs[index]! - anchor.x) * 256 * dt) * damping
    anchor.vy     = (anchor.vy + (ys[index]! - anchor.y) * 256 * dt) * damping
    anchor.x     += anchor.vx * dt
    anchor.y     += anchor.vy * dt
  }

  /**
   * Moves each link's reach a frame on, out toward its bird or back to the
   * picked bird where its bird has gone or the links are hidden, taking up its
   * next bird once it is home. A link setting out while none shows starts below
   * zero by its bird's rank, so the seven reach out one after another, nearest
   * first, whereas while the drawing stands still each link is at once fully
   * out or fully home.
   */
  reachOn(dt: number, out: boolean, still: boolean): void {
    let showing = false
    for (let link = 0; link < common.NEIGHBORS; link++) showing ||= this.#reach[link]! > 0
    for (let link = 0; link < common.NEIGHBORS; link++) {
      if (out && this.#bird[link]! >= 0 && this.#next[link]! < 0) {
        this.#reach[link] = still ? 1 : Math.min(1, this.#reach[link]! + dt * REACH.out)
        continue
      }
      this.#reach[link] = still ? 0 : Math.max(0, this.#reach[link]! - dt * REACH.back)
      if (this.#reach[link]! > 0) continue
      const bird       = out ? this.#next[link]! : -1
      this.#bird[link] = bird
      this.#next[link] = -1
      if (bird < 0) continue
      this.#reach[link] = still ? 1 : showing ? 0 : -REACH.stagger * this.#rank(bird)
    }
  }

  /**
   * Lets every link go and hides them, as when the drawing is fitted afresh.
   */
  reset(): void {
    this.#bird.fill(-1)
    this.#next.fill(-1)
    this.#reach.fill(0)
    this.anchor.w = 0
  }
}
