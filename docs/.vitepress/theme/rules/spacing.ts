import { Slider, Toggle, markup }              from '@thermur/kit'
import { defineComponent, h, ref, shallowRef } from 'vue'

import { useStage }         from '../../lib/canvas/mount.ts'
import { TRIALS, rulesOf }  from '../../lib/rules/measured.ts'
import { NAMES, type Rail } from '../../lib/rules/spacing.ts'
import { Sweep, spoken }    from '../../lib/rules/sweep.ts'

/**
 * The polarization each coupling rule holds against the launch spacing, read
 * from the trials `mise measure` last wrote out, set on a track a reader drags
 * or plays. A slider stands over the track drawn into the chart and a toggle
 * over the knob that plays it, so a drag moves the chart at once and a release
 * or an arrow key glides it to a measured spacing.
 */
export default defineComponent({
  name: 'RulesSpacing',
  setup() {
    const rules   = rulesOf(NAMES, TRIALS.coupling)
    const stops   = rules[0]!.points.map(point => point.given)
    const playing = ref(false)
    const rail    = shallowRef<Pick<Rail, 'player' | 'track'>>()
    const value   = ref(stops[0]!)
    let sweep: Sweep | undefined
    const host    = useStage(box => {
      sweep = new Sweep(box, rules, {
        moved  : moved => value.value = Math.round(moved * 10) / 10,
        placed : placed => rail.value = placed,
        played : on => playing.value = on
      })
      return sweep
    })
    return () => h('figure', { class: 'rules-figure rules-over' }, [
      h('div', { ref: host }),
      rail.value && [
        h(Slider, {
          class    : 'kit-slider-over',
          label    : 'Launch spacing',
          max      : stops.at(-1)!,
          min      : stops[0]!,
          spoken   : (spacing: number) => spoken(stops, spacing),
          step     : 0.1,
          stops,
          style    : markup.placed(rail.value.track),
          value    : value.value,
          onChange : (spacing: number) => sweep?.drag(spacing),
          onCommit : (spacing: number) => sweep?.settle(spacing)
        }),
        h(Toggle, {
          class    : 'rules-play',
          label    : 'Play the sweep across launch spacings',
          pressed  : playing.value,
          style    : markup.placed(rail.value.player),
          onChange : () => sweep?.play()
        }, { default: () => [] })
      ]
    ])
  }
})
