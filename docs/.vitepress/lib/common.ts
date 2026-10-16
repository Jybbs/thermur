/**
 * A rectangle, in pixels across and down from a top left corner, from `x0` to
 * `x1` across and from `y0` to `y1` down.
 */
export interface Box {
  x0 : number
  x1 : number
  y0 : number
  y1 : number
}

/**
 * Two numbers, a point's distance across and down from the top left corner of
 * what it sits on, or the least and the most of a range.
 */
export type Pair = readonly [number, number]

/**
 * A ridge across a drawing, `base` pixels down where its three waves cancel,
 * rising and falling up to `amplitude` around it, each wave repeating every one
 * of `periods` pixels and starting at one of `phases`.
 */
interface Ridge {
  amplitude : number
  base      : number
  periods   : readonly [number, number, number]
  phases    : readonly [number, number, number]
}

/**
 * The number of nearest neighbors each bird reads, the topological count the
 * starling studies measured.
 */
export const NEIGHBORS = 7

/**
 * Reads the value `map` holds under `key`, building it through `build` and
 * storing it there on the first read.
 */
export function cached<K, V>(map: Map<K, V>, key: K, build: () => V): V {
  let value = map.get(key)
  if (value === undefined) {
    value = build()
    map.set(key, value)
  }
  return value
}

/**
 * Reads the red, the green, and the blue of a `#rrggbb` hex.
 */
export const channels = (hex: string): number[] =>
  [1, 3, 5].map(start => Number.parseInt(hex.slice(start, start + 2), 16))

/**
 * Holds `value` between `from` and `to`, between 0 and 1 unless given.
 */
export const clamp = (value: number, from = 0, to = 1): number => Math.min(to, Math.max(from, value))

/**
 * Reads the height at `x` of the ridge whose waves of `amplitude` cancel at
 * `base`, its waves weighted 0.55, 0.3, and 0.15 from the longest to the
 * shortest.
 */
export function heightAt({ amplitude, base, periods, phases }: Ridge, x: number): number {
  return base - amplitude * (
      0.55 * Math.sin(x / periods[0] + phases[0])
    + 0.3 * Math.sin(x / periods[1] + phases[1])
    + 0.15 * Math.sin(x / periods[2] + phases[2])
  )
}

/**
 * Lays `share` of the color `over` across the color `base`, channel by channel
 * in sRGB, the share held between 0 and 1.
 */
export function mix(base: string, over: string, share: number): string {
  const [from, to] = [channels(base), channels(over)]
  const weight     = clamp(share)
  const mixed      = from.map((channel, index) => Math.round(channel + (to[index]! - channel) * weight))
  return `#${mixed.map(channel => channel.toString(16).padStart(2, '0')).join('')}`
}

/**
 * Builds the Mulberry32 generator from `seed`, whose every call returns the
 * next number of one fixed sequence in [0, 1), so a scene drawn from the same
 * seed is drawn alike every time.
 */
export function mulberry(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state     = (state + 0x6d2b79f5) >>> 0
    let mixed = Math.imul(state ^ (state >>> 15), state | 1)
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61)
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Writes the nearest neighbors of the point at `index` into `into` from `base`
 * on, nearest first, as many as `best` holds, the points placed at `xs` and
 * `ys`, `best` holding the squared distances found so far.
 */
export function nearest(
  best  : Float32Array,
  index : number,
  into  : Int16Array | Int32Array,
  xs    : ArrayLike<number>,
  ys    : ArrayLike<number>,
  base  : number = 0
): void {
  const last = best.length - 1
  best.fill(Infinity)
  for (let other = 0; other < xs.length; other++) {
    const distance = (xs[other]! - xs[index]!) ** 2 + (ys[other]! - ys[index]!) ** 2
    if (other === index || distance >= best[last]!) continue
    let k = last
    for (; k > 0 && best[k - 1]! > distance; k--) {
      best[k]        = best[k - 1]!
      into[base + k] = into[base + k - 1]!
    }
    best[k]        = distance
    into[base + k] = other
  }
}

/**
 * Builds the height of `ridge` at each point across a drawing, as `heightAt`
 * reads it.
 */
export const ridgeline = (ridge: Ridge): ((x: number) => number) => x => heightAt(ridge, x)

/**
 * Eases from 0 to 1 as `x` runs from `from` to `to`, flat outside them.
 */
export function smoothstep(from: number, to: number, x: number): number {
  const t = clamp((x - from) / (to - from))
  return t * t * (3 - 2 * t)
}
