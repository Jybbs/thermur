import { defineComponent, h } from 'vue'

import { useStage }        from '../../lib/canvas/mount.ts'
import { NAMES }           from '../../lib/rules/dial.ts'
import { TRIALS, rulesOf } from '../../lib/rules/measured.ts'
import { Reach }           from '../../lib/rules/reach.ts'

/**
 * The angle the farthest drones turn through under each heading rule against
 * the turn the first drone makes, read from the trials `mise measure` last
 * wrote out, on a quarter dial whose turn a reader picks on its rim.
 */
export default defineComponent({
  name: 'RulesReach',
  setup() {
    const host = useStage(box => new Reach(box, rulesOf(NAMES, TRIALS.heading)))
    return () => h('figure', { class: 'rules-figure', ref: host })
  }
})
