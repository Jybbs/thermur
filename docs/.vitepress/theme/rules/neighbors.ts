import { defineComponent, h } from 'vue'

import { useStage }  from '../../lib/canvas/mount.ts'
import { Neighbors } from '../../lib/rules/neighbors.ts'

/**
 * The neighbors a drone senses, by rank against within a fixed range: the
 * front page's flock under the pointer's falcon, one bird marked with its
 * seven and its range, beneath a header of the controls and over two strips
 * counting each over the last seconds.
 */
export default defineComponent({
  name: 'RulesNeighbors',
  setup() {
    const host = useStage(box => new Neighbors(box))
    return () => h('figure', { class: 'rules-neighbors', ref: host })
  }
})
