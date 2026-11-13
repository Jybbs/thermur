import { Chip, Radios, Slider }    from '@thermur/kit'
import { defineComponent, h, ref } from 'vue'

import { useStage }                       from '../../lib/canvas/mount.ts'
import { MODES, type Mode }               from '../../lib/rules/falcon.ts'
import { FALCON, MODE, Neighbors, RANGE } from '../../lib/rules/neighbors.ts'

/**
 * The radios choosing the pointer's mode, each beside its icon and titled with
 * what the pointer does in it.
 */
const OPTIONS = MODES.map(mode => ({ title: MODE[mode].title, value: mode, words: MODE[mode].words }))

/**
 * The neighbors a drone senses, by rank against within a fixed range: the
 * front page's flock under the pointer's falcon, one bird marked with its
 * seven and its range, beneath a header naming the strips beside the radios
 * choosing the pointer's mode, the slider setting the fixed range, and the
 * chip reading how far out the marked bird's seventh flies, over two strips
 * counting each over the last seconds.
 */
export default defineComponent({
  name: 'RulesNeighbors',
  setup() {
    const mode  = ref<Mode>(MODES[0])
    const range = ref<number>(RANGE.value)
    const named = ref<HTMLElement>()
    const value = ref<HTMLElement>()
    let drawing: Neighbors | undefined
    const host  = useStage(box => {
      drawing = new Neighbors(box, { named: named.value!, value: value.value! })
      return drawing
    })
    return () => h('figure', { class: 'rules-neighbors' }, [
      h('div', { class: 'rules-bar' }, [
        h('span', { class: 'rules-title' }, `Birds around the marked bird, last ${FALCON.trace} s`),
        h(Radios, {
          label    : 'Pointer',
          options  : OPTIONS,
          value    : mode.value,
          onChange : (picked: string) => {
            mode.value = picked as Mode
            drawing?.mode(mode.value)
          }
        }, {
          item: ({ value: picked }: { value: string }) => h('svg', { 'aria-hidden': 'true', 'viewBox': '0 0 16 16' },
            h('path', { d: MODE[picked as Mode].icon }))
        }),
        h('span', { class: 'rules-range' }, [
          h('span', { 'aria-hidden': 'true', 'class': 'kit-slider-label' }, 'Range'),
          h(Slider, {
            label    : 'Range, in bird lengths',
            max      : RANGE.max,
            min      : RANGE.min,
            step     : RANGE.step,
            value    : range.value,
            onChange : (lengths: number) => {
              range.value = lengths
              drawing?.range(lengths)
            }
          })
        ]),
        h(Chip, { name: 'Seventh by rank', named, value })
      ]),
      h('div', { class: 'rules-stage', ref: host })
    ])
  }
})
