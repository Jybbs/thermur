/**
 * A traced bird as an SVG scene draws it: its fill, its turn in degrees, and
 * where its head sits.
 */
interface Drawn {
  fill : string
  turn : number
  x    : number
  y    : number
}

/**
 * Reads each traced bird an SVG scene draws, in the order the scene draws them.
 */
export const birds = (svg: string): Drawn[] =>
  [...svg.matchAll(/<use href="#g\d+" fill="(#[\da-f]{6})" transform="translate\(([\d.-]+) ([\d.-]+)\) rotate\(([\d.-]+)\)/gu)]
    .map(([, fill, x, y, turn]) => ({ fill: fill!, turn: Number(turn), x: Number(x), y: Number(y) }))
