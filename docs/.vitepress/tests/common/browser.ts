import { type Mock, vi } from 'vitest'

/**
 * The stand-ins a drawing's suite runs on in place of the browser's own, each
 * counting what is still held, so a case can check that a teardown left nothing
 * behind.
 */
export interface Browser {
  built   : () => Built
  context : Record<string, Mock>
  fonts   : { load: Mock<(font: string) => Promise<FontFace[]>> }
  frames  : Frames
  held    : () => Held
  motion  : { listeners: Set<() => void>, matches: boolean }
  mutate  : () => void
  pen     : CanvasRenderingContext2D
  reflow  : () => void
  resize  : (width: number, height: number) => void
  see     : (seen: boolean) => void
  wrote   : (key: string) => unknown[]
}

/**
 * What a drawing has built since the case began: the elements the document
 * created, and the gradients and paths the canvas built.
 */
interface Built {
  elements  : number
  gradients : number
  paths     : number
}

/**
 * The animation frames a case runs by hand through `run`, which runs every
 * scheduled frame `count` times, `step` milliseconds apart.
 */
interface Frames {
  run : (count: number, step?: number) => void
}

/**
 * What a drawing still holds: the canvases in the document, the animation
 * frames scheduled, the listeners live on the window, the root element, and any
 * canvas, the reduced-motion listeners, and the observers watching.
 */
interface Held {
  canvases  : number
  frames    : number
  listeners : number
  motion    : number
  observed  : number
}

/**
 * A pointer's place as a pointer event carries it, in the client's coordinates
 * or in the offset from its target.
 */
type Place = Partial<Record<'clientX' | 'clientY' | 'offsetX' | 'offsetY', number>>

/**
 * What the stand-in context measures any text at: ink 200 pixels across, rising
 * 120 pixels above its baseline and dropping 4 below it.
 */
const METRICS = {
  actualBoundingBoxAscent  : 120,
  actualBoundingBoxDescent : 4,
  actualBoundingBoxLeft    : 100,
  actualBoundingBoxRight   : 100
} as const

/**
 * An observer stand-in of one `kind`, which counts itself held while it watches
 * any target and holds each target it watches.
 */
class Watcher {
  static readonly active = new Set<Watcher>()
  readonly callback : (entries: unknown[]) => void
  readonly kind     : string
  readonly targets  = new Set<Element>()

  constructor(kind: string, callback: (entries: unknown[]) => void) {
    this.kind     = kind
    this.callback = callback
  }

  disconnect(): void {
    this.targets.clear()
    Watcher.active.delete(this)
  }

  observe(target: Element): void {
    this.targets.add(target)
    Watcher.active.add(this)
  }
}

/**
 * What a drawing holds once its teardown has released everything.
 */
export const RELEASED: Held = { canvases: 0, frames: 0, listeners: 0, motion: 0, observed: 0 }

/**
 * Reports whether a listener added with `options` was released through the
 * abort signal it was added with.
 */
const aborted = (options: unknown): boolean =>
  (options as AddEventListenerOptions | undefined)?.signal?.aborted === true

/**
 * Installs a 2D context whose every method is a spy, measuring any text at
 * `METRICS`, and whose every property set is written down in order, a `Path2D`
 * stand-in, and a canvas measuring 1200 by 800 pixels on the page until a case
 * resizes it and nothing once off the page, counting each element, gradient,
 * and path built.
 */
function canvas(): Pick<Browser, 'built' | 'context' | 'pen' | 'resize' | 'wrote'> {
  let gradients                      = 0
  let paths                          = 0
  let size                           = { height: 800, width: 1200 }
  const written: [string, unknown][] = []
  const gradient                     = (): { addColorStop: Mock } => {
    gradients++
    return { addColorStop: vi.fn<() => void>() }
  }
  const context: Record<string, Mock> = {
    createLinearGradient : vi.fn<typeof gradient>(gradient),
    createRadialGradient : vi.fn<typeof gradient>(gradient),
    measureText          : vi.fn<() => Partial<TextMetrics>>(() => METRICS)
  }
  const proxy = new Proxy(context, {
    get : (held, key: string) => held[key] ??= vi.fn<() => void>(),
    set : (_, key: string, value: unknown) => written.push([key, value]) > 0
  })
  const { prototype } = HTMLCanvasElement
  const made          = vi.spyOn(document, 'createElement')
  vi.spyOn(prototype, 'getContext').mockReturnValue(proxy as unknown as CanvasRenderingContext2D)
  vi.spyOn(prototype, 'getBoundingClientRect').mockImplementation(function measured(this: HTMLCanvasElement) {
    return (this.isConnected ? size : { height: 0, width: 0 }) as DOMRect
  })
  vi.stubGlobal('Path2D', function Path2D() {
    paths++
    return { closePath: (): void => {}, lineTo: (): void => {}, moveTo: (): void => {} }
  })
  return {
    built   : () => ({ elements: made.mock.calls.length, gradients, paths }),
    context : context,
    pen     : proxy as unknown as CanvasRenderingContext2D,
    wrote   : key => written.filter(([set]) => set === key).map(([, value]) => value),
    resize  : (width, height) => {
      size = { height, width }
    }
  }
}

/**
 * Installs animation frames a case runs by hand on a clock of its own.
 */
function frames(): Frames & { pending: Map<number, FrameRequestCallback> } {
  const pending = new Map<number, FrameRequestCallback>()
  let   next    = 0
  let   clock   = 0
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    pending.set(++next, callback)
    return next
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => pending.delete(id))
  return {
    pending : pending,
    run     : (count, step = 16) => {
      for (let k = 0; k < count; k++) {
        const due = [...pending.values()]
        pending.clear()
        clock += step
        for (const callback of due) callback(clock)
      }
    }
  }
}

/**
 * Counts the listeners still live on the window, the root element, and any
 * canvas, each one added and neither removed nor released through its abort
 * signal.
 */
function listeners(): () => number {
  const spied = [window, document.documentElement, HTMLCanvasElement.prototype].map(target => ({
    added   : vi.spyOn(target, 'addEventListener'),
    removed : vi.spyOn(target, 'removeEventListener')
  }))
  return () => spied.flatMap(({ added, removed }) => added.mock.calls.filter(([type, listener, options]) =>
    !aborted(options)
    && !removed.mock.calls.some(([gone, held]) => gone === type && held === listener))).length
}

/**
 * Calls back each observer stand-in of `kind` that watches, or that watches
 * `target` where one is given, with `entries`.
 */
function notify(kind: string, entries: unknown[], target?: Element): void {
  for (const watcher of Watcher.active) {
    const watching = target === undefined || watcher.targets.has(target)
    if (watcher.kind === kind && watching) watcher.callback(entries)
  }
}

/**
 * Writes `before` and then a box of the class `name` into the document's body,
 * and returns the box.
 */
export function box(name: string, before = ''): HTMLElement {
  document.body.innerHTML = `${before}<div class="${name}"></div>`
  return document.querySelector<HTMLElement>(`.${name}`)!
}

/**
 * Installs the stand-ins: a canvas, observers that count what they watch,
 * animation frames a case runs by hand, a reduced-motion query the case sets,
 * listeners counted on the window, the root element, and any canvas, and a font
 * set whose every face loads at once, each but the font set restored once the
 * case ends through the suite's `unstubGlobals` and `restoreMocks`, and the
 * font set replaced by the next case's.
 *
 * Args:
 *   reduced: Whether the reader prefers reduced motion.
 */
export function browser(reduced = false): Browser {
  Watcher.active.clear()
  for (const kind of ['IntersectionObserver', 'MutationObserver', 'ResizeObserver']) {
    vi.stubGlobal(kind, function Observer(callback: (entries: unknown[]) => void) {
      return new Watcher(kind, callback)
    })
  }
  const fonts  = { load: vi.fn<(font: string) => Promise<FontFace[]>>(() => Promise.resolve([])) }
  const motion = { listeners: new Set<() => void>(), matches: reduced }
  const live   = listeners()
  const drawn  = canvas()
  const queued = frames()
  Object.defineProperty(document, 'fonts', { configurable: true, value: fonts })
  vi.stubGlobal('matchMedia', () => ({
    get matches() { return motion.matches },

    addEventListener: (_: string, listener: () => void, options?: AddEventListenerOptions) => {
      motion.listeners.add(listener)
      options?.signal?.addEventListener('abort', () => motion.listeners.delete(listener))
    }
  }))
  return {
    ...drawn,
    fonts  : fonts,
    frames : queued,
    motion : motion,
    mutate : () => notify('MutationObserver', []),
    reflow : () => notify('ResizeObserver', [], document.body),
    see    : seen => notify('IntersectionObserver', [{ isIntersecting: seen }]),
    held   : () => ({
      canvases  : document.querySelectorAll('canvas').length,
      frames    : queued.pending.size,
      listeners : live(),
      motion    : motion.listeners.size,
      observed  : Watcher.active.size
    })
  }
}

/**
 * Moves the pointer over `target` to the place `at` holds, written as the
 * client or the offset coordinates a pointer event carries.
 */
export function point(target: EventTarget, at: Place): void {
  target.dispatchEvent(Object.assign(new Event('pointermove'), at))
}
