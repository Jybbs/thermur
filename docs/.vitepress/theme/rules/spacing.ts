import { defineComponent, h } from 'vue'

import { useStage }        from '../../lib/canvas/mount.ts'
import { TRIALS, rulesOf } from '../../lib/rules/measured.ts'
import { NAMES }           from '../../lib/rules/spacing.ts'
import { Sweep }           from '../../lib/rules/sweep.ts'

/**
 * The polarization each coupling rule holds against the launch spacing, read
 * from the trials `mise measure` last wrote out, set on a track a reader drags
 * or plays.
 */
export default defineComponent({
  name: 'RulesSpacing',
  setup() {
    const host = useStage(box => new Sweep(box, rulesOf(NAMES, TRIALS.coupling)))
    return () => h('figure', { class: 'rules-figure', ref: host })
  }
})
