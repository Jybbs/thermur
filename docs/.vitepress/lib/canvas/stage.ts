import type { Scheme }         from '../identity.ts'
import type { Geometry, Part } from './part.ts'
import { paused }              from './pause.ts'

/**
 * Where a stage's canvas sits, `pinned` behind the whole page and reading the
 * pointer across the page in its coordinates, or else in the page's flow and
 * reading the pointer over the canvas in its own.
 */
interface Mount {
  pinned : boolean
}

/**
 * The observers a stage holds: one watching its canvas come on and off the
 * screen, one watching the root element's scheme and pause, and one watching
 * its canvas and the page's body change size.
 */
interface Observers {
  intersection : IntersectionObserver
  mutation     : MutationObserver
  resize       : ResizeObserver
}

/**
 * The longest step a stalled frame counts as, in seconds.
 */
const STALL = 0.05

/**
 * A canvas, which fits its parts to the page, steps them on the browser's
 * frames while the canvas is on screen and motion runs, paints them in the
 * order given, and hands each a click on the canvas. Motion stands still for a
 * reader who prefers reduced motion and while the page's pause is on, where the
 * stage paints one frame and then again only on a scroll, a pointer, a click, a
 * change of scheme, or a reflow of the page. `destroy` removes every listener,
 * observer, and frame the stage holds, together with its canvas.
 */
export class Stage {
  readonly #abort    = new AbortController()
  readonly #canvas   : HTMLCanvasElement
  readonly #client   : [number, number] = [0, 0]
  readonly #context  : CanvasRenderingContext2D
  #dpr               = 1
  #frame             = 0
  readonly #g        : Geometry
  #last              = 0
  readonly #motion   : MediaQueryList
  readonly #observed : Observers
  readonly #parts    : readonly Part[]
  readonly #pinned   : boolean
  readonly #pointer  : [number, number] = [0, 0]
  #running           = false
  #size              = { h: 0, page: 0, w: 0 }
  #visible           = false

  constructor(host: HTMLElement, parts: readonly Part[], { pinned }: Mount) {
    this.#parts   = parts
    this.#pinned  = pinned
    this.#canvas  = document.createElement('canvas')
    this.#context = this.#canvas.getContext('2d')!
    this.#motion  = window.matchMedia('(prefers-reduced-motion: reduce)')
    this.#g       = {
      doc     : null,
      dy      : 0,
      h       : 0,
      hero    : { bottom: 0, h: 0, top: 0 },
      page    : 0,
      pointer : null,
      scheme  : this.#scheme,
      still   : false,
      t       : 0,
      w       : 0
    }
    host.append(this.#canvas)
    this.#observed = {
      intersection : new IntersectionObserver(this.#onSeen),
      mutation     : new MutationObserver(this.#onRoot),
      resize       : new ResizeObserver(this.#onResize)
    }
    this.#observed.intersection.observe(this.#canvas)
    this.#observed.mutation.observe(document.documentElement, { attributeFilter: ['class', 'data-paused'] })
    this.#observed.resize.observe(this.#canvas)
    this.#observed.resize.observe(document.body)
    const { signal }                 = this.#abort
    const moves: GlobalEventHandlers = pinned ? window : this.#canvas
    const leaves                     = pinned ? document.documentElement : this.#canvas
    this.#motion.addEventListener('change', this.#onRoot, { signal })
    window.addEventListener('scroll', this.#onScroll, { passive: true, signal })
    moves.addEventListener('pointermove', this.#onPointer, { passive: true, signal })
    leaves.addEventListener('pointerleave', this.#onLeave, { signal })
    this.#canvas.addEventListener('click', this.#onPress, { signal })
    this.#onResize()
  }

  /**
   * Reads the page's length, the longer of the window and the page itself.
   */
  get #pageHeight(): number {
    return Math.max(window.innerHeight, document.documentElement.scrollHeight)
  }

  /**
   * Reads the scheme the site's appearance switch has set.
   */
  get #scheme(): Scheme {
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
  }

  /**
   * Reads whether the drawing stands still, for a reader who prefers reduced
   * motion or while the page's pause is on.
   */
  get #still(): boolean {
    return this.#motion.matches || paused()
  }

  /**
   * Updates the record the parts read to this frame and returns it, placing the
   * pointer of a pinned stage on the page as it is now scrolled.
   */
  #geometry(): Geometry {
    const g          = this.#g
    g.w              = this.#size.w
    g.h              = this.#size.h
    g.dy             = window.scrollY
    g.page           = this.#size.page
    g.scheme         = this.#scheme
    g.still          = this.#still
    this.#pointer[0] = this.#client[0]
    this.#pointer[1] = this.#client[1] + (this.#pinned ? g.dy : 0)
    return g
  }

  /**
   * Starts the frames where the canvas is on screen, motion runs, and none is
   * already scheduled.
   */
  #kick(): void {
    if (this.#running || !this.#visible || this.#still) return
    this.#running = true
    this.#last    = 0
    this.#frame   = requestAnimationFrame(this.#tick)
  }

  /**
   * Measures where the hero and the reading column stand on the page, which
   * moves only when the page is resized or reflows.
   */
  #measure(): void {
    const g       = this.#g
    const hero    = document.querySelector('.VPHero')?.getBoundingClientRect()
    const column  = document.querySelector('.VPHomeContent')?.getBoundingClientRect()
    g.hero.top    = (hero?.top ?? 0) + window.scrollY
    g.hero.h      = hero?.height ?? 0
    g.hero.bottom = g.hero.top + g.hero.h
    g.doc         = column === undefined ? null : {
      bottom : column.bottom + window.scrollY,
      left   : column.left,
      right  : column.right,
      top    : column.top + window.scrollY
    }
  }

  /**
   * Clears the pointer once it leaves the page, or the canvas for a stage in
   * the page's flow, repainting where the drawing stands still.
   */
  readonly #onLeave = (): void => {
    this.#g.pointer = null
    if (this.#still) this.#paint()
  }

  /**
   * Holds the pointer's place, in the window for a pinned stage and on the
   * canvas for one in the page's flow, repainting where the drawing stands
   * still.
   */
  readonly #onPointer = (event: PointerEvent): void => {
    this.#client[0] = this.#pinned ? event.clientX : event.offsetX
    this.#client[1] = this.#pinned ? event.clientY : event.offsetY
    this.#g.pointer = this.#pointer
    if (this.#still) this.#paint()
  }

  /**
   * Hands a click on the canvas to each part that takes one, repainting where
   * the drawing stands still.
   */
  readonly #onPress = (): void => {
    const g = this.#geometry()
    for (const part of this.#parts) part.press?.(g)
    if (this.#still) this.#paint()
  }

  /**
   * Fits the canvas to its box at the screen's pixel ratio, up to two, and the
   * parts to the page, once on mounting and again whenever the canvas or the
   * page's body changes size.
   */
  readonly #onResize = (): void => {
    const { height, width } = this.#canvas.getBoundingClientRect()
    this.#dpr               = Math.min(2, window.devicePixelRatio)
    this.#canvas.width      = Math.round(width * this.#dpr)
    this.#canvas.height     = Math.round(height * this.#dpr)
    this.#refit(Math.round(width), Math.round(height))
  }

  /**
   * Repaints for a changed scheme and starts or holds the frames for a changed
   * pause or motion preference.
   */
  readonly #onRoot = (): void => {
    this.#paint()
    this.#kick()
  }

  /**
   * Repaints as the page scrolls where the drawing stands still.
   */
  readonly #onScroll = (): void => {
    if (this.#still) this.#paint()
  }

  /**
   * Starts or stops the frames as the canvas comes on and off the screen.
   */
  readonly #onSeen = (entries: IntersectionObserverEntry[]): void => {
    this.#visible = entries.at(-1)!.isIntersecting
    this.#kick()
  }

  /**
   * Paints every part onto a cleared canvas.
   */
  #paint(): void {
    if (this.#size.w === 0) return
    const ctx = this.#context
    const g   = this.#geometry()
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height)
    ctx.setTransform(this.#dpr, 0, 0, this.#dpr, 0, 0)
    for (const part of this.#parts) part.paint(ctx, g, this.#dpr)
  }

  /**
   * Fits every part to a canvas `w` by `h` pixels and the page's length, then
   * paints them.
   */
  #refit(w = this.#size.w, h = this.#size.h): void {
    if (w === 0 || h === 0) return
    this.#size = { h, page: this.#pageHeight, w }
    this.#measure()
    const g = this.#geometry()
    for (const part of this.#parts) part.fit?.(g)
    this.#paint()
  }

  /**
   * Steps and paints the parts on one frame and schedules the next, a stalled
   * frame counting as `STALL` seconds at most, or stops where the canvas left
   * the screen or motion stood still.
   */
  readonly #tick = (now: number): void => {
    if (!this.#visible || this.#still) {
      this.#running = false
      return
    }
    const dt   = this.#last === 0 ? 1 / 60 : Math.min(STALL, (now - this.#last) / 1000)
    this.#last = now
    this.#g.t += dt
    const g    = this.#geometry()
    for (const part of this.#parts) part.step?.(dt, g)
    this.#paint()
    this.#frame = requestAnimationFrame(this.#tick)
  }

  /**
   * Removes every listener, observer, and frame the stage holds, and its
   * canvas.
   */
  destroy(): void {
    cancelAnimationFrame(this.#frame)
    this.#observed.intersection.disconnect()
    this.#observed.mutation.disconnect()
    this.#observed.resize.disconnect()
    this.#abort.abort()
    this.#canvas.remove()
  }

  /**
   * Fits the canvas and every part again, as once the faces a part measures
   * have loaded, leaving a canvas no longer on the page unfitted.
   */
  refit(): void {
    this.#onResize()
  }
}
