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
 * Does nothing, standing in for what stops a transition while none runs.
 */
export const idle = (): void => {}

/**
 * Runs `step` with the eased share of a transition `scale` times the theme's
 * settling time long, or once with 1 where that time is zero, and returns
 * what stops it. A first frame stamped before the transition began takes no
 * share at all.
 */
export function tween(step: (share: number) => void, scale = 1): () => void {
  const length = settle() * scale
  if (length <= 0) {
    step(1)
    return idle
  }
  const start = performance.now()
  let handle  = 0
  const frame = (now: number): void => {
    const share = Math.min(1, Math.max(0, now - start) / length)
    step(eased(share))
    handle      = share < 1 ? requestAnimationFrame(frame) : 0
  }
  handle      = requestAnimationFrame(frame)
  return () => cancelAnimationFrame(handle)
}
