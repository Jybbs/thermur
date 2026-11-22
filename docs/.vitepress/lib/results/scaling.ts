import swept from '../measured/still-air.json' with { type: 'json' }

import { type Identity, type Scheme, fontStack, seriesOf, shadeOf } from '../identity.ts'
import source                                                     from '../rendered/identity.json' with { type: 'json' }

/**
 * One size's statistic as a figure reads it: the mean across seeds beside its
 * standard error, both `null` where no seed measured it.
 */
interface Point {
  error : number | null
  mean  : number | null
  size  : number
}

/**
 * A norm as a figure draws it: its figure, the spread about it, and whether
 * the source states only that the statistic exceeds the figure.
 */
interface Norm {
  above  : boolean
  figure : number
  spread : number | null
}

/**
 * One figure: the statistic of the composite's flights and of its fixed-speed
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
    const line  = points.map(point => `${x(point.size).toFixed(1)},${y(point.mean!).toFixed(1)}`).join(' ')
    const bars  = points.map(({ error, mean, size }) =>
      `<line x1="${x(size).toFixed(1)}" x2="${x(size).toFixed(1)}" y1="${y(mean! - (error ?? 0)).toFixed(1)}" y2="${y(mean! + (error ?? 0)).toFixed(1)}" stroke="${color}" stroke-width="2"/>`
      + `<circle cx="${x(size).toFixed(1)}" cy="${y(mean!).toFixed(1)}" r="4" fill="${color}"/>`)
    return `<polyline points="${line}" fill="none" stroke="${color}" stroke-width="2"${place ? ' stroke-dasharray="6 4"' : ''}/>${bars.join('')}`
  })
  const band = norm === undefined ? ''
    : norm.spread === null
      ? `<line x1="${left}" x2="${width - right}" y1="${y(norm.figure).toFixed(1)}" y2="${y(norm.figure).toFixed(1)}" stroke="${ink}" stroke-opacity="0.5" stroke-dasharray="2 4"/>`
      : `<rect x="${left}" y="${y(bounds[1]!).toFixed(1)}" width="${width - left - right}" height="${(y(bounds[0]!) - y(bounds[1]!)).toFixed(1)}" fill="${ink}" fill-opacity="0.1"/>`
  const tick = (attributes: string, words: string) =>
    `<text ${attributes} fill="${ink}" font-family="${fontStack(faces.mono)}" font-size="12">${words}</text>`
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${chart.title}">`,
    `<rect width="${width}" height="${height}" fill="${shadeOf('ground', scheme, identity)}"/>`,
    `<text x="${left}" y="28" fill="${ink}" font-family="${fontStack(faces.display)}" font-size="20">${chart.title}</text>`,
    band,
    ...ticks(...range).map(value =>
      `<line x1="${left}" x2="${width - right}" y1="${y(value).toFixed(1)}" y2="${y(value).toFixed(1)}" stroke="${ink}" stroke-opacity="0.12"/>`
      + tick(`x="${left - 8}" y="${(y(value) + 4).toFixed(1)}" text-anchor="end"`, String(value))),
    ...sizes.map(size => tick(`x="${x(size).toFixed(1)}" y="${height - bottom + 20}" text-anchor="middle"`, size.toLocaleString('en-US'))),
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
