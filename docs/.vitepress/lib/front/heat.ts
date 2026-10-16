/**
 * A source of heat a flock flies through, giving the heat at a point.
 */
export interface Heat {
  heat : (x: number, y: number) => number
}

/**
 * Writes into `out` the unit direction `field`'s heat rises toward at a point,
 * from central differences two pixels either side.
 */
function uphill(field: Heat, x: number, y: number, out: [number, number]): void {
  const gx     = field.heat(x + 2, y) - field.heat(x - 2, y)
  const gy     = field.heat(x, y + 2) - field.heat(x, y - 2)
  const length = Math.hypot(gx, gy) + 1e-6
  out[0]       = gx / length
  out[1]       = gy / length
}

/**
 * Writes into `out` the push away from `field`'s heat at a point, down its
 * slope at `strength` once the heat `hot` there reaches 0.42, and in proportion
 * to it below that.
 */
export function repel(
  field    : Heat,
  x        : number,
  y        : number,
  hot      : number,
  strength : number,
  out      : [number, number]
): void {
  uphill(field, x, y, out)
  const push = -strength * Math.min(1, hot * 2.4)
  out[0]    *= push
  out[1]    *= push
}
