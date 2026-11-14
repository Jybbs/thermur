import { Select }                                        from '@thermur/kit'
import { type VNode, computed, defineComponent, h, ref } from 'vue'

import { useStage }   from '../lib/canvas/mount.ts'
import type { Stage } from '../lib/canvas/stage.ts'
import { drawKept }   from '../lib/crop/drawing.ts'
import { Kept }       from '../lib/crop/kept.ts'
import * as sentence  from '../lib/crop/sentence.ts'

/**
 * How much of an output the graph reads, set through three menus inside one
 * sentence, under a strip sketching the part kept and over the lines the picks
 * write. Each menu stands at the width of its widest choice, so no pick moves
 * the words around it.
 */
export default defineComponent({
  name: 'CropKept',
  setup() {
    const kept  = new Kept()
    const menus = sentence.menusOf(kept)
    const picks = ref(menus.map(() => 0))
    const lines = computed(() => sentence.linesOf(menus, picks.value))
    let stage: Stage | undefined
    const host  = useStage(box => {
      stage = drawKept(box, kept)
      return stage
    })
    const menu  = (index: number): VNode => {
      const { choices, label } = menus[index]!
      return h(Select, {
        choices  : choices.map(({ name }) => name),
        label,
        picked   : picks.value[index]!,
        onChange : (place: number) => {
          choices[place]!.apply()
          picks.value[index] = place
          stage?.repaint()
        }
      })
    }
    return () => h('figure', { class: 'crop-kept' }, [
      h('div', { 'aria-hidden': 'true', 'class': 'crop-canvas', 'ref': host }),
      h('div', { class: 'crop-sentence' }, [
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
