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
