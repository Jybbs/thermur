import { defineComponent, h, useId } from 'vue'

import credit from '../../lib/rendered/credit.json' with { type: 'json' }

/**
 * The band of snapshots in the pixels of its viewbox: each dot's radius `dot`,
 * and the `step` from one dot to the next.
 */
const BAND = { dot: 2.6, step: 7 } as const

/**
 * The sample's snapshots as one band of dots in the cut's own line, warming
 * through the heat ramp from the first time to the last.
 */
export default defineComponent({
  name: 'CreditsSnapshots',
  setup() {
    const fill  = useId()
    const width = credit.snapshots * BAND.step
    return () => h('svg', {
      'aria-hidden' : 'true',
      'class'       : 'credits-snapshots',
      'style'       : { aspectRatio: `${width} / ${BAND.step}` },
      'viewBox'     : `0 0 ${width} ${BAND.step}`
    }, [
      h('defs', [
        h('linearGradient', { gradientUnits: 'userSpaceOnUse', id: fill, x1: 0, x2: width },
          ['cool', 'mid', 'hot'].map((tone, place) => h('stop', { class: `credits-${tone}`, offset: place / 2 })))
      ]),
      ...Array.from({ length: credit.snapshots }, (_, place) =>
        h('circle', { cx: (place + 0.5) * BAND.step, cy: BAND.step / 2, fill: `url(#${fill})`, r: BAND.dot }))
    ])
  }
})
