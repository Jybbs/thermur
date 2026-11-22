import swept from '../measured/still-air.json' with { type: 'json' }

import { type Identity, type Scheme, fontStack, seriesOf, shadeOf } from '../identity.ts'
import source                                                     from '../rendered/identity.json' with { type: 'json' }

/**
 * One size's statistic as a figure reads it, meaning the mean across seeds beside its
 * standard error, both `null` where no seed measured it.
 */
interface Point {
  error : number | null
  mean  : number | null
  size  : number
}

/**
 * A norm as a figure draws it, meaning its figure, the spread about it, and whether
 * the source states only that the statistic exceeds the figure.
 */
interface Norm {
  above  : boolean
  figure : number
  spread : number | null
}

/**
 * One figure, holding the statistic of the composite's flights and of its fixed-speed
 * control by size, the norm it is set beside where one is declared, its title,
 * and the label of its vertical axis.
 */
export interface Scaling {
  fixed    : readonly Point[]
  flocking : readonly Point[]
  label    : string
  title    : string
  norm?    : Norm
}

/**
 * The width and height a figure is drawn at, in pixels, the text column's
 * width, beside the room each side leaves for the title, the ticks, and the
 * labels.
 */
const FRAME = { bottom: 56, height: 360, left: 72, right: 24, top: 48, width: 688 } as const

/**
 * The words each figure sets for the statistic it draws, by the statistic's
 * name in the sweep.
 */
const WORDS: Readonly<Record<string, Pick<Scaling, 'label' | 'title'>>> = {
  aspect_ratio            : { label: 'Longest Axis over Thinnest', title: 'Aspect Ratio by Flock Size' },
  border_density          : { label: 'Border Density over Inside', title: 'Border Density by Flock Size' },
  connected               : { label: 'Share in the Largest Radio Group', title: 'Radio Connectivity by Flock Size' },
  group_speed_m_s         : { label: 'Group Speed (m/s)', title: 'Group Speed by Flock Size' },
  middle_aspect_ratio     : { label: 'Middle Axis over Thinnest', title: 'Middle Aspect Ratio by Flock Size' },
  polarization            : { label: 'Polarization', title: 'Polarization by Flock Size' },
  spacing_m               : { label: 'Nearest-Neighbor Distance (m)', title: 'Spacing by Flock Size' },
  speed_correlation_ratio : { label: 'Correlation Length over Extent', title: 'Speed Correlation Length by Flock Size' },
  wave_speed_m_s          : { label: 'Turning Wave Speed (m/s)', title: 'Turning Wave Speed by Flock Size' }
}

/**
 * Writes a coordinate to one decimal place, the precision a figure's SVG
 * keeps.
 */
const at = (value: number) => value.toFixed(1)

/**
 * Lays out between four and ten ticks across `low` to `high` at a step of one,
 * two, or five times a power of ten.
 */
export function ticks(low: number, high: number): number[] {
  const rough = (high - low) / 4
  const power = 10 ** Math.floor(Math.log10(rough))
  const step  = [1, 2, 5, 10].map(each => each * power).find(each => each >= rough)!
  const first = Math.ceil(low / step)
  return Array.from({ length: Math.floor(high / step) - first + 1 }, (_, index) => +((first + index) * step).toPrecision(12))
}

/**
 * Draws one figure as SVG in `scheme`: the composite's mean by size on a
 * logarithmic axis joined by a line, its control's dashed, each with its
 * standard error as a bar, over a band across the norm's spread or a dashed
 * line at a figure stated alone, and a size no seed measured left out.
 */
export function scaling(chart: Scaling, scheme: Scheme, identity: Identity = source): string {
  const { bottom, height, left, right, top, width } = FRAME
  const kept   = [chart.flocking, chart.fixed].map(points => points.filter(point => point.mean !== null))
  const values = kept.flat().flatMap(({ error, mean }) => [mean! - (error ?? 0), mean! + (error ?? 0)])
  const norm   = chart.norm
  const bounds = norm === undefined ? [] : [norm.figure - (norm.spread ?? 0), norm.figure + (norm.spread ?? 0)]
  const span   = [Math.min(...values, ...bounds), Math.max(...values, ...bounds)] as const
  const pad    = (span[1] - span[0] || Math.abs(span[0]) || 1) * 0.08
  const range  = [span[0] - pad, span[1] + pad] as const
  const sizes  = chart.flocking.map(point => point.size)
  const logs   = [Math.log(Math.min(...sizes)), Math.log(Math.max(...sizes))] as const
  const x      = (size: number) => left + (Math.log(size) - logs[0]) / (logs[1] - logs[0]) * (width - left - right)
  const y      = (value: number) => height - bottom - (value - range[0]) / (range[1] - range[0]) * (height - top - bottom)
  const ink    = shadeOf('ink', scheme, identity)
  const faces  = identity.faces
  const series = kept.map((points, place) => {
    const color = seriesOf(place + 1, scheme, identity)
    const line  = points.map(point => `${at(x(point.size))},${at(y(point.mean!))}`).join(' ')
    const bars  = points.map(({ error, mean, size }) =>
      `<line x1="${at(x(size))}" x2="${at(x(size))}" y1="${at(y(mean! - (error ?? 0)))}" y2="${at(y(mean! + (error ?? 0)))}" stroke="${color}" stroke-width="2"/>`
      + `<circle cx="${at(x(size))}" cy="${at(y(mean!))}" r="4" fill="${color}"/>`)
    return `<polyline points="${line}" fill="none" stroke="${color}" stroke-width="2"${place ? ' stroke-dasharray="6 4"' : ''}/>${bars.join('')}`
  })
  const across = (value: number, attributes: string) =>
    `<line x1="${left}" x2="${width - right}" y1="${at(y(value))}" y2="${at(y(value))}" stroke="${ink}" ${attributes}/>`
  const band = norm === undefined ? ''
    : norm.spread === null
      ? across(norm.figure, 'stroke-opacity="0.5" stroke-dasharray="2 4"')
      : `<rect x="${left}" y="${at(y(bounds[1]!))}" width="${width - left - right}" height="${at(y(bounds[0]!) - y(bounds[1]!))}" fill="${ink}" fill-opacity="0.1"/>`
  const tick = (attributes: string, words: string) =>
    `<text ${attributes} fill="${ink}" font-family="${fontStack(faces.mono)}" font-size="12">${words}</text>`
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${chart.title}">`,
    `<rect width="${width}" height="${height}" fill="${shadeOf('ground', scheme, identity)}"/>`,
    `<text x="${left}" y="28" fill="${ink}" font-family="${fontStack(faces.display)}" font-size="20">${chart.title}</text>`,
    band,
    ...ticks(...range).map(value =>
      across(value, 'stroke-opacity="0.12"')
      + tick(`x="${left - 8}" y="${at(y(value) + 4)}" text-anchor="end"`, String(value))),
    ...sizes.map(size => tick(`x="${at(x(size))}" y="${height - bottom + 20}" text-anchor="middle"`, size.toLocaleString('en-US'))),
    `<text x="${(left + width - right) / 2}" y="${height - 12}" fill="${ink}" font-family="${fontStack(faces.body)}" font-size="14" text-anchor="middle">Flock Size (drones, logarithmic)</text>`,
    `<text transform="translate(18 ${(top + height - bottom) / 2}) rotate(-90)" fill="${ink}" font-family="${fontStack(faces.body)}" font-size="14" text-anchor="middle">${chart.label}</text>`,
    ...series,
    '</svg>'
  ].join('\n') + '\n'
}

/**
 * Each figure The Flock in Still Air embeds, by the statistic it draws, read
 * from the sweep `mise measure` last wrote out, beside its norm where the
 * sweep's file carries one.
 */
export const FIGURES: Readonly<Record<string, Scaling>> = Object.fromEntries(
  Object.entries(WORDS).map(([name, words]) => {
    const statistic = swept.statistics[name as keyof typeof swept.statistics]
    const norm      = (swept.norms as Readonly<Record<string, Norm>>)[name]
    return [name, { ...words, fixed: statistic.fixed, flocking: statistic.flocking, ...norm && { norm } }]
  })
)
