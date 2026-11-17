/**
 * The media query a reader's reduced-motion preference matches.
 */
const REDUCED = '(prefers-reduced-motion: reduce)'

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
 * Turns the page's pause on or off, which stands every drawing still while it
 * is on.
 */
export const pause = (on: boolean): void => {
  document.documentElement.dataset.paused = String(on)
}

/**
 * Reads whether the page's pause is on.
 */
export const paused = (): boolean => document.documentElement.dataset.paused === 'true'

/**
 * Reads whether motion that starts on its own stands still, for a reader who
 * prefers reduced motion or while the page's pause is on.
 */
export const still = (): boolean => paused() || matchMedia(REDUCED).matches
