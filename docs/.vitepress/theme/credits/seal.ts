import { type VNode, defineComponent, h, useId } from 'vue'

import { GROUND, SEAL, rimOf, rimPath } from '../../lib/credits/seal.ts'

/**
 * Lays out the site's short fire at the seal's middle, a hairline of flame on
 * the ground under its glow, which the gradient `glow` names paints.
 */
function flame(glow: string): VNode[] {
  const { center, flame: { glow: reach, half, rise, y } } = SEAL
  const crest = `M${center - half} ${y}Q${center} ${y - 2 * rise} ${center + half} ${y}`
  return [
    h('path', { d: `M${center - reach} ${y}a${reach} ${reach} 0 0 1 ${2 * reach} 0z`, fill: `url(#${glow})` }),
    h('path', { class: 'credits-ground', d: `M${center - GROUND} ${y}h${2 * GROUND}` }),
    h('path', { class: 'credits-flame-ember', d: crest }),
    h('path', { class: 'credits-flame', d: crest })
  ]
}

/**
 * The round seal the Data page's credits stand beside, its rim reading the
 * plume dataset's creators, its year, and its license around the site's own
 * flame.
 */
export default defineComponent({
  name: 'CreditsSeal',
  setup() {
    const id                                        = useId()
    const { center, flame: { glow, y }, rim, size } = SEAL
    const { foot, top }                             = rimOf()
    const arc  = (name: 'foot' | 'top', words: string): VNode[] => [
      h('path', { d: rimPath(rim[name], name === 'top'), fill: 'none', id: `${id}-${name}` }),
      h('text', { class: 'credits-rim' }, [
        h('textPath', { href: `#${id}-${name}`, startOffset: '50%' }, words)
      ])
    ]
    const ring = (part: 'core' | 'edge' | 'hair'): VNode =>
      h('circle', { class: `credits-ring-${part}`, cx: center, cy: center, r: SEAL[part] })
    return () => h('svg', {
      'aria-hidden' : 'true',
      'class'       : 'credits-seal',
      'viewBox'     : `0 0 ${size} ${size}`
    }, [
      h('defs', [
        h('radialGradient', {
          cx            : center,
          cy            : y,
          gradientUnits : 'userSpaceOnUse',
          id            : `${id}-glow`,
          r             : glow
        }, [0, 0.5, 1].map(offset => h('stop', { class: 'credits-glow', offset })))
      ]),
      ring('edge'),
      ring('hair'),
      ring('core'),
      ...flame(`${id}-glow`),
      ...arc('top', top),
      ...arc('foot', foot),
      ...[-1, 1].map(side => h('circle', { class: 'credits-dot', cx: center + side * rim.side, cy: center, r: rim.dot }))
    ])
  }
})
