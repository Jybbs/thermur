import { Radios, markup }                      from '@thermur/kit'
import { defineComponent, h, ref, shallowRef } from 'vue'

import { useStage }        from '../../lib/canvas/mount.ts'
import { NAMES }           from '../../lib/rules/dial.ts'
import { TRIALS, rulesOf } from '../../lib/rules/measured.ts'
import { Reach }           from '../../lib/rules/reach.ts'

/**
 * The angle the farthest drones turn through under each heading rule against
 * the turn the first drone makes, read from the trials `mise measure` last
 * wrote out, on a quarter dial whose turn a reader picks with the radios
 * standing over the stops on its rim.
 */
export default defineComponent({
  name: 'RulesReach',
  setup() {
    const rules  = rulesOf(NAMES, TRIALS.heading)
    const turns  = rules[0]!.points.map(point => point.given)
    const picked = ref(0)
    const stops  = shallowRef<readonly markup.Spot[]>([])
    let reach: Reach | undefined
    const host   = useStage(box => {
      reach = new Reach(box, spots => stops.value = spots, rules)
      return reach
    })
    return () => h('figure', { class: 'rules-figure rules-over' }, [
      h('div', { ref: host }),
      h(Radios, {
        class    : 'kit-radios-over',
        label    : 'The first drone turns',
        options  : turns.map((turn, place) => ({
          style : stops.value[place] && markup.placed(stops.value[place]),
          value : String(place),
          words : `${turn}°`
        })),
        value    : String(picked.value),
        onChange : (value: string) => {
          picked.value = Number(value)
          reach?.go(picked.value)
        }
      })
    ])
  }
})
