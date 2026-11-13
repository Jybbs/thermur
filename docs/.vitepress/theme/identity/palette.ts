import { type Copier, heard, useCopy }                   from '@thermur/kit'
import { useData }                                       from 'vitepress'
import { type Ref, type VNode, defineComponent, h, ref } from 'vue'

import * as identity from '../../lib/identity.ts'
import source        from '../../lib/rendered/identity.json' with { type: 'json' }
import * as sketch   from '../../lib/specimen/sketch.ts'
import * as parts    from './parts.ts'

/**
 * What a press runs to copy the hex `pair` takes in the scheme on display.
 */
type Copying = (pair: Schemed) => (event: Event) => Promise<void>

/**
 * A color each scheme takes, by the scheme.
 */
type Schemed = identity.Identity['palette']['series'][number]

/**
 * The side of the swatch beside each level's name, in the chart's units.
 */
const SWATCH = 6

/**
 * Writes where a mark of the seconds along the chart's foot anchors its text,
 * the first and the last at their ends and the rest at their middles.
 */
const anchored = (second: number): string =>
  second === 0 ? 'start' : second === sketch.SPAN ? 'end' : 'middle'

/**
 * Lays out the flight chart: its frame, the warning and critical lines, and a
 * line per chart series, each drawn on while `shown` holds it.
 */
function chart(shown: readonly boolean[]): VNode {
  const { height, width } = sketch.CHART
  return h('svg', {
    'aria-label' : 'Temperatures through a plume, one line per chart series, against the warning and '
                 + 'critical lines',
    'class'      : 'identity-chart',
    'role'       : 'img',
    'viewBox'    : `0 0 ${width} ${height}`
  }, [
    ...frame(),
    level('warning'),
    level('critical'),
    ...shown.map((on, place) => h('path', {
      'class'      : 'identity-line',
      'data-shown' : String(on),
      'd'          : sketch.seriesPath(place),
      'pathLength' : 1,
      'style'      : { stroke: `var(--th-series-${place + 1})` }
    }))
  ])
}

/**
 * Builds what a press runs to copy a pair's hex in the scheme `dark` reads.
 */
const copying = (copy: Copier['copy'], dark: Ref<boolean>): Copying => pair => event =>
  copy(identity.colorOf(pair[dark.value ? 'dark' : 'light']), event)

/**
 * Lays out the flight chart's frame: its ground, the temperatures up its side,
 * the seconds along its foot, and the line beneath the flight.
 */
function frame(): VNode[] {
  const { height, width } = sketch.CHART
  return [
    h('rect', { class: 'identity-plate', height, rx: 14, width }),
    ...sketch.KELVINS.map(kelvin =>
      h('text', {
        'class'       : 'identity-quiet',
        'text-anchor' : 'end',
        'x'           : 28,
        'y'           : sketch.chartY(kelvin) + 3
      }, kelvin)),
    ...sketch.SECONDS.map(second =>
      h('text', {
        'class'       : 'identity-quiet',
        'text-anchor' : anchored(second),
        'x'           : sketch.chartX(second),
        'y'           : height - 10
      }, `${second} s`)),
    h('line', {
      class : 'identity-axis',
      x1    : sketch.chartX(0),
      x2    : sketch.chartX(sketch.SPAN),
      y1    : height - 26,
      y2    : height - 26
    })
  ]
}

/**
 * Lays out the hex of the color each scheme takes from `pair`, the stylesheet
 * showing the one of the scheme on display, so the page the server writes
 * matches every scheme.
 */
const hexOf = (pair: Schemed): VNode =>
  h('span', { class: 'identity-hex kit-fuse' }, [
    h('span', { class: 'identity-on-light' }, identity.colorOf(pair.light)),
    h('span', { class: 'identity-on-dark' }, identity.colorOf(pair.dark))
  ])

/**
 * Lays out the line the flight chart rules across at `role`'s level, its stroke
 * in the color of the role over an outline in ink, and the role's name in ink
 * beside a swatch of that color ringed in ink.
 */
function level(role: keyof typeof sketch.LEVELS): VNode {
  const [left, right] = [sketch.chartX(0), sketch.chartX(sketch.SPAN)]
  const [color, ink]  = [`var(--th-${role})`, 'var(--vp-c-text-1)']
  const y             = sketch.chartY(sketch.LEVELS[role])
  return h('g', { class: 'identity-level' }, [
    h('line', { class: 'identity-level-edge', style: { stroke: ink }, x1: left, x2: right, y1: y, y2: y }),
    h('line', { class: 'identity-level-line', style: { stroke: color }, x1: left, x2: right, y1: y, y2: y }),
    h('rect', {
      class  : 'identity-level-swatch',
      height : SWATCH,
      style  : { fill: color, stroke: ink },
      width  : SWATCH,
      x      : right - SWATCH,
      y      : y - 5 - SWATCH
    }),
    h('text', {
      'class'       : 'identity-mono',
      'style'       : { fill: ink },
      'text-anchor' : 'end',
      'x'           : right - SWATCH - 4,
      'y'           : y - 5
    }, role)
  ])
}

/**
 * Lays out the key of palette role `role`, a press on which copies its hex.
 */
const roleKey = (copied: Copying, role: identity.Role): VNode =>
  parts.keyOf(role, `var(--th-${role})`, hexOf(source.palette[role]), copied(source.palette[role]))

/**
 * Lays out the card of the chart series at `place`: a switch drawing its line
 * on or lifting it off, beside a press copying its hex.
 */
function seriesCard(copied: Copying, place: number, shown: Ref<boolean[]>): VNode {
  const pair = source.palette.series[place]!
  return h('div', { class: 'identity-series' }, [
    h('button', {
      'aria-checked' : String(shown.value[place]),
      'class'        : 'identity-switch',
      'onClick'      : () => (shown.value[place] = !shown.value[place]),
      'role'         : 'switch',
      'type'         : 'button'
    }, [
      h('svg', { 'aria-hidden': 'true', 'class': 'identity-spark', 'viewBox': '0 0 60 20' }, [
        h('path', { d: sketch.sparkPath(place), style: { stroke: `var(--th-series-${place + 1})` } })
      ]),
      h('span', { class: 'identity-caps' }, `Chart ${place + 1}`)
    ]),
    h('button', { class: 'identity-copy', onClick: copied(pair), type: 'button' }, hexOf(pair))
  ])
}

/**
 * The palette at work: a sketch of temperatures through a plume drawn in each
 * chart series against the warning and critical lines, a switch per series that
 * draws its line on or lifts it off beside a press that copies its hex, and a
 * key per role that copies the role's hex, each hex the one of the scheme on
 * display.
 */
export default defineComponent({
  name: 'IdentityPalette',
  setup() {
    const { isDark }     = useData()
    const { copy, said } = useCopy()
    const copied         = copying(copy, isDark)
    const shown          = ref(source.palette.series.map(() => true))
    return () => h('figure', { class: 'identity-palette' }, [
      chart(shown.value),
      parts.madeUp('its temperatures'),
      h('div', { class: 'identity-series-list' },
        shown.value.map((_, place) => seriesCard(copied, place, shown))),
      h('div', { class: 'identity-keys' }, identity.roles.map(role => roleKey(copied, role))),
      heard(said)
    ])
  }
})
