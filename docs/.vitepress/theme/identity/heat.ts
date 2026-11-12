import { type VNode, defineComponent, h, useId } from 'vue'

import { heatRamp }             from '../../lib/identity.ts'
import source                   from '../../lib/rendered/identity.json' with { type: 'json' }
import { type Copier, useCopy } from '../../lib/specimen/copy.ts'
import * as sketch              from '../../lib/specimen/sketch.ts'
import * as parts               from './parts.ts'

/**
 * Lays out the labels at the heat profile's edges: the temperature at either
 * end of its foot and the height at its top.
 */
function edges(): VNode[] {
  const { height, kelvins: [cool, hot], rise } = sketch.PROFILE
  return [
    h('text', { 'class': 'identity-quiet', 'x': sketch.profileX(cool), 'y': height - 8 }, `${cool} K`),
    h('text', {
      'class'       : 'identity-quiet',
      'text-anchor' : 'end',
      'x'           : sketch.profileX(hot),
      'y'           : height - 8
    }, `${hot} K`),
    h('text', { 'class': 'identity-quiet', 'text-anchor': 'end', 'x': 26, 'y': 18 }, `${rise} m`)
  ]
}

/**
 * Lays out the readout beneath the heat profile: the height the knot sits at,
 * the air's temperature there, and the step of the ramp it takes, beside a
 * chip that copies the step's hex.
 */
function readout(copy: Copier['copy'], steps: readonly string[]): VNode {
  const { knot } = sketch.PROFILE
  const kelvin   = sketch.warmthAt(knot)
  const place    = sketch.stepAt(kelvin, steps.length)
  const hex      = steps[place]!
  return h('div', { class: 'identity-readout' }, [
    h('span', { class: 'identity-caps' }, `${knot} m · ${Math.round(kelvin)} K · Step ${place + 1}`),
    h('button', { class: 'identity-chip', onClick: (event: Event) => copy(hex, event), type: 'button' }, [
      h('i', { style: { background: `var(--th-heat-${place + 1})` } }),
      h('span', { class: 'identity-hex' }, hex)
    ])
  ])
}

/**
 * The heat ramp at work: a sketch of the air's temperature over height above a
 * fire, filled in the step of the ramp each temperature takes, its knot fixed
 * at one height over a readout of the air there and a chip that copies the
 * step's hex.
 */
export default defineComponent({
  name: 'IdentityHeat',
  setup() {
    const { copy, said }          = useCopy()
    const { height, knot, width } = sketch.PROFILE
    const fill                    = useId()
    const steps                   = heatRamp(source)
    return () => h('figure', { class: 'identity-heat' }, [
      h('svg', {
        'aria-label' : 'The air’s temperature over height above a fire, filled in the step of the heat '
                     + 'ramp each temperature takes',
        'class'      : 'identity-profile',
        'role'       : 'img',
        'viewBox'    : `0 0 ${width} ${height}`
      }, [
        h('defs', [
          h('linearGradient', { id: fill, x1: 0, x2: 1 }, steps.map((_, place) =>
            h('stop', {
              offset : `${place / (steps.length - 1) * 100}%`,
              style  : { stopColor: `var(--th-heat-${steps.length - place})` }
            })))
        ]),
        h('rect', { class: 'identity-plate', height, rx: 12, width }),
        h('path', { d: sketch.profileArea(), fill: `url(#${fill})` }),
        h('path', { class: 'identity-profile-line', d: sketch.profilePath() }),
        ...edges(),
        h('circle', {
          class : 'identity-knot',
          cx    : sketch.profileX(sketch.warmthAt(knot)),
          cy    : sketch.profileY(knot),
          r     : 4.5
        })
      ]),
      parts.madeUp('its temperatures'),
      readout(copy, steps),
      parts.heard(said)
    ])
  }
})
