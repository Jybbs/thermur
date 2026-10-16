import type { Plugin } from 'vite'

import source from './rendered/identity.json' with { type: 'json' }

/**
 * The identity as `docs:render` writes it out of `thermur.identity`.
 */
export type Identity = typeof source

/**
 * What a custom property the theme sets paints, which decides the contrast it
 * holds: a text role holds 4.5:1 against every ground, a mark 3:1, the ink
 * holds 4.5:1 against every fill, the surface of a control, and a rule, which
 * only divides one surface from another, holds none.
 */
type Kind = 'fill' | 'ground' | 'mark' | 'rule' | 'text'

/**
 * A custom property the theme sets to a share of one of the palette's roles
 * laid over another, the ground unless `over` names a role, under the kind of
 * thing it paints.
 */
export interface Mix {
  kind  : Kind
  role  : Role
  share : number
  over? : Role
}

/**
 * A palette entry holding one color per scheme rather than a list.
 */
type Role = {
  [K in keyof Identity['palette']]: Identity['palette'][K] extends { dark: string, light: string } ? K : never
}[keyof Identity['palette']]

/**
 * One of the two color schemes each role holds a color for.
 */
export type Scheme = keyof Identity['palette']['ground']

/**
 * The kind of text each custom property's face is set for, the default theme's
 * own for the words and the code, and the site's for its headings.
 */
const FACES = {
  '--th-font-display'     : 'display',
  '--vp-font-family-base' : 'body',
  '--vp-font-family-mono' : 'mono'
} as const satisfies Record<string, keyof Identity['faces']>
/**
 * The default theme's custom properties the site sets, each to a share of a
 * palette role laid over the ground, or over the role its `over` names, so each
 * one follows the site's own appearance switch wherever its role differs
 * between the schemes.
 */
export const MIXES: Readonly<Record<string, Mix>> = {
  '--vp-c-bg'           : { kind: 'ground', role: 'ground', share: 1 },
  '--vp-c-bg-alt'       : { kind: 'ground', role: 'ink', share: 0.03 },
  '--vp-c-bg-elv'       : { kind: 'ground', role: 'ground', share: 1 },
  '--vp-c-bg-soft'      : { kind: 'ground', role: 'ink', share: 0.05 },
  '--vp-c-border'       : { kind: 'rule', role: 'ink', share: 0.24 },
  '--vp-c-brand-1'      : { kind: 'text', role: 'lead', share: 1 },
  '--vp-c-brand-2'      : { kind: 'text', over: 'ink', role: 'lead', share: 0.8 },
  '--vp-c-brand-3'      : { kind: 'mark', role: 'lead', share: 1 },
  '--vp-c-brand-soft'   : { kind: 'fill', role: 'lead', share: 0.14 },
  '--vp-c-default-1'    : { kind: 'fill', role: 'ink', share: 0.16 },
  '--vp-c-default-2'    : { kind: 'fill', role: 'ink', share: 0.12 },
  '--vp-c-default-3'    : { kind: 'fill', role: 'ink', share: 0.08 },
  '--vp-c-default-soft' : { kind: 'ground', role: 'ink', share: 0.05 },
  '--vp-c-divider'      : { kind: 'rule', role: 'ink', share: 0.12 },
  '--vp-c-gutter'       : { kind: 'rule', role: 'ink', share: 0.12 },
  '--vp-c-text-1'       : { kind: 'text', role: 'ink', share: 1 },
  '--vp-c-text-2'       : { kind: 'text', role: 'ink', share: 0.78 },
  '--vp-c-text-3'       : { kind: 'text', role: 'ink', share: source.tagline.alpha }
}
/**
 * The id the theme imports the identity's stylesheet by.
 */
const MODULE = 'virtual:identity.css'
/**
 * The default theme's custom properties for the label of a button filled in
 * the lead, pressed, under the pointer, and at rest, each beside the property
 * the theme fills that button in. Each label is set in the ground, which holds
 * a text's contrast against its fill in both schemes.
 */
export const ON_LEAD = {
  '--vp-button-brand-active-text' : '--vp-c-brand-1',
  '--vp-button-brand-hover-text'  : '--vp-c-brand-2',
  '--vp-button-brand-text'        : '--vp-c-brand-3'
} as const

/**
 * Writes the color of a role holding one per scheme, as its hex where the two
 * schemes share it and through `light-dark()` where they differ.
 */
const schemed = (role: { dark: string, light: string }): string =>
  role.dark === role.light ? colorOf(role.light) : `light-dark(${colorOf(role.light)}, ${colorOf(role.dark)})`

/**
 * Reads the hex of the palette color filed under `name`.
 *
 * Raises:
 *   Error: Where the palette declares no color under `name`.
 */
export function colorOf(name: string, identity: Identity = source): string {
  const hex = (identity.palette.colors as Record<string, string>)[name]
  if (hex === undefined) throw new Error(`the palette declares no color named ${name}`)
  return hex
}

/**
 * Writes the font stack `face` names in CSS, its family quoted and then its
 * fallbacks.
 */
export const fontStack = (face: Pick<Identity['faces']['body'], 'fallbacks' | 'family'>): string =>
  [`'${face.family}'`, ...face.fallbacks].join(', ')

/**
 * Writes the address of the Google Fonts stylesheet that loads every face at
 * the weights the identity sets, roman and italic where a face takes both.
 */
export function fontsHref(identity: Identity = source): string {
  const families = Object.values(identity.faces).map(face => {
    const styles = face.italic ? [0, 1] : [0]
    const pairs  = styles.flatMap(style => face.weights.map(weight => `${style},${weight}`))
    const axes   = styles.length > 1 ? `:ital,wght@${pairs.join(';')}` : `:wght@${face.weights.join(';')}`
    return `family=${face.family.replaceAll(' ', '+')}${pairs.join() === '0,400' ? '' : axes}`
  })
  return `https://fonts.googleapis.com/css2?${families.join('&')}&display=swap`
}

/**
 * Reads the hex of each step of the heat ramp, from light to dark.
 */
export const heatRamp = (identity: Identity): string[] =>
  identity.palette.heat.map(name => colorOf(name, identity))

/**
 * Writes the identity's stylesheet, declaring each palette role, each chart
 * series, and each step of the heat ramp as a custom property, the two colors
 * of a role differing between the schemes through `light-dark()`, then every
 * property `MIXES` names and every face's stack.
 */
export function identityCss(): string {
  const { palette } = source
  const roles       = (Object.keys(palette) as (keyof typeof palette)[])
    .filter((key): key is Role => 'dark' in palette[key])
    .toSorted()
    .map(role => `  --th-${role}: ${schemed(palette[role])};`)
  const series = palette.series.map((role, index) => `  ${seriesProperty(index)}: ${schemed(role)};`)
  const heat   = heatRamp(source).map((hex, index) => `  --th-heat-${index + 1}: ${hex};`)
  const mixes  = Object.entries(MIXES).map(([property, { over = 'ground', role, share }]) =>
    share === 1
      ? `  ${property}: var(--th-${role});`
      : `  ${property}: color-mix(in srgb, var(--th-${role}) ${Math.round(share * 100)}%, `
      + `var(--th-${over}));`)
  const labels = Object.keys(ON_LEAD).map(property => `  ${property}: var(--th-ground);`)
  const faces  = Object.entries(FACES).map(([property, kind]) =>
    `  ${property}: ${fontStack(source.faces[kind])};`)

  return `:root {\n${[...roles, ...series, ...heat, ...mixes, ...labels, ...faces].join('\n')}\n}\n`
}

/**
 * Serves the identity's stylesheet as a virtual module the theme imports, so it
 * joins the stylesheet the build writes and passes through Lightning CSS with
 * the rest. The id it resolves to opens on a null byte, the convention Vite's
 * plugins keep for a module no file stands behind.
 */
export function identityModule(): Plugin {
  return {
    load      : id => id === `\0${MODULE}` ? identityCss() : undefined,
    name      : 'thermur:identity',
    resolveId : id => id === MODULE ? `\0${MODULE}` : undefined
  }
}

/**
 * Reads the hex of the chart series color at `place` in `scheme`, the series
 * color at place 0 being the fire's.
 *
 * Raises:
 *   Error: Where the palette holds no series color at `place`.
 */
export function seriesOf(place: number, scheme: Scheme, identity: Identity = source): string {
  const pair = identity.palette.series[place]
  if (pair === undefined) throw new Error(`the palette holds no series color at place ${place}`)
  return colorOf(pair[scheme], identity)
}

/**
 * Names the custom property the identity's stylesheet sets to the chart series
 * color at `index` of the palette's series, the fire's at index 0.
 */
export const seriesProperty = (index: number): string => `--th-series-${index + 1}`

/**
 * Reads the hex of palette role `role` in `scheme`.
 */
export const shadeOf = (role: Role, scheme: Scheme, identity: Identity = source): string =>
  colorOf(identity.palette[role][scheme], identity)
