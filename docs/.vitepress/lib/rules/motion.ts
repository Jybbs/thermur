import { paused } from '../canvas/pause.ts'

/**
 * The media query a reader's reduced-motion preference matches.
 */
const REDUCED = '(prefers-reduced-motion: reduce)'

/**
 * Eases a share of a transition fast out of the start and slow into the end,
 * near the theme's settling curve.
 */
const eased = (share: number): number => 1 - (1 - share) ** 3

/**
 * Reads the length of one transition, in milliseconds, from the theme's shared
 * `--th-settle`, which the theme sets to zero under a reduced-motion
 * preference.
 */
function settle(): number {
  const value  = getComputedStyle(document.documentElement).getPropertyValue('--th-settle').trim()
  const length = Number(value.replace(/m?s$/u, ''))
  if (!Number.isFinite(length)) return 0
  return value.endsWith('ms') ? length : length * 1000
}

/**
 * Calls `changed` whenever the page's pause switch or the reduced-motion
 * preference changes, returning what releases both.
 */
export function onMotion(changed: () => void): () => void {
  const abort = new AbortController()
  const root  = new MutationObserver(changed)
  root.observe(document.documentElement, { attributeFilter: ['data-paused'] })
  matchMedia(REDUCED).addEventListener('change', changed, { signal: abort.signal })
  return () => {
    root.disconnect()
    abort.abort()
  }
}

/**
 * Reads whether motion that starts on its own stands still, for a reader who
 * prefers reduced motion or while the page's pause is on.
 */
export const still = (): boolean => paused() || matchMedia(REDUCED).matches

/**
 * Runs `step` with the eased share of a transition `scale` times the theme's
 * settling time long, once with 1 where that time is zero, and returns what
 * stops it.
 */
export function tween(step: (share: number) => void, scale = 1): () => void {
  const length = settle() * scale
  if (length <= 0) {
    step(1)
    return () => {}
  }
  const start = performance.now()
  let handle  = 0
  const frame = (now: number): void => {
    const share = Math.min(1, (now - start) / length)
    step(eased(share))
    handle      = share < 1 ? requestAnimationFrame(frame) : 0
  }
  handle      = requestAnimationFrame(frame)
  return () => cancelAnimationFrame(handle)
}

/**
 * Calls `changed` with the host's width, rounded to a pixel, whenever the host
 * changes to a width it has not been drawn at, returning what releases the
 * observer.
 */
export function widthOf(host: HTMLElement, changed: (width: number) => void): () => void {
  let drawn      = 0
  const observer = new ResizeObserver(() => {
    const width = Math.round(host.getBoundingClientRect().width)
    if (width === drawn || width === 0) return
    drawn = width
    changed(width)
  })
  observer.observe(host)
  return () => observer.disconnect()
}
