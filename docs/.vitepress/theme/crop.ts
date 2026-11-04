import { type VNode, computed, defineComponent, h, ref } from 'vue'

import { useStage }   from '../lib/canvas/mount.ts'
import type { Stage } from '../lib/canvas/stage.ts'
import { drawKept }   from '../lib/crop/drawing.ts'
import { Kept }       from '../lib/crop/kept.ts'
import * as sentence  from '../lib/crop/sentence.ts'

/**
 * How much of an output the graph reads, set through three menus inside one
 * sentence, under a strip sketching the part kept and over the lines the picks
 * write.
 */
export default defineComponent({
  name: 'CropKept',
  setup() {
    const kept  = new Kept()
    const menus = sentence.menusOf(kept)
    const picks = ref(menus.map(() => 0))
    const lines = computed(() => sentence.linesOf(menus, picks.value))
    const words = sentence.useHeldHeight()
    let stage: Stage | undefined
    const host  = useStage(box => {
      stage = drawKept(box, kept)
      return stage
    })
    const menu  = (index: number): VNode => {
      const { choices, label } = menus[index]!
      const onChange           = (event: Event): void => {
        const { selectedIndex } = event.target as HTMLSelectElement
        choices[selectedIndex]!.apply()
        picks.value[index] = selectedIndex
        stage?.repaint()
      }
      return h('select', { 'aria-label': label, onChange }, choices.map(({ name }) => h('option', name)))
    }
    return () => h('figure', { class: 'crop-kept' }, [
      h('div', { 'aria-hidden': 'true', 'class': 'crop-canvas', 'ref': host }),
      h('p', { class: 'crop-sentence', ref: words }, [
        'Read ', menu(0), ' for ', menu(1), `, from ${kept.model.floor_m} m up to `, menu(2), '.'
      ]),
      h('div', {
        'aria-live'  : 'polite',
        'class'      : 'crop-lines',
        'data-empty' : lines.value === null ? '' : undefined
      }, lines.value ?? sentence.DEFAULTS)
    ])
  }
})
