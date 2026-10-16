/**
 * An oval a stream parts around, reaching `a` pixels across and `b` pixels down
 * from its middle at `cx` and `cy`.
 */
export interface Oval {
  a  : number
  b  : number
  cx : number
  cy : number
}

/**
 * Finds the height at which a bird keeping to the line it held far upstream,
 * `far` pixels down, passes the column `x` pixels across. The height comes from
 * the stream function of a steady flow around a cylinder, stretched to `oval`:
 * the bird's line keeps its value of the stream function, and halving finds the
 * height outside the oval where the stream function takes that value at this
 * column.
 */
export function passing(x: number, far: number, oval: Oval): number {
  const across = (x - oval.cx) / oval.a
  const target = (far - oval.cy) / oval.b
  const square = across * across
  const goal   = Math.abs(target)
  let low      = Math.sqrt(Math.max(0, 1 - square))
  let high     = goal + 2
  for (let halving = 0; halving < 22; halving++) {
    const middle = (low + high) / 2
    if (middle - middle / (square + middle * middle) < goal) low = middle
    else high = middle
  }
  return oval.cy + (target < 0 ? -1 : 1) * oval.b * (low + high) / 2
}

/**
 * Measures how far outside `oval` the point at `x` and `y` stands, 1 on its rim
 * and less inside it.
 */
export const rimOf = (oval: Oval, x: number, y: number): number =>
  Math.hypot((x - oval.cx) / oval.a, (y - oval.cy) / oval.b)
