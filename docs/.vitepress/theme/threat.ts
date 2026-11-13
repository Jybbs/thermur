import { Chip, Slider } from '@thermur/kit'
import { type ComponentPublicInstance, type VNode, defineComponent, h, ref } from 'vue'

import { useStage }   from '../lib/canvas/mount.ts'
import type { Stage } from '../lib/canvas/stage.ts'
import { drawThreat } from '../lib/threat/drawing.ts'
import { Front }      from '../lib/threat/front.ts'
import { Playback }   from '../lib/threat/playback.ts'
import { spoken }     from '../lib/threat/readings.ts'

/**
 * The marks the drawing's key names, by the class giving each its form.
 */
const KEY = [
  { mark: 'now', name: 'Air now' },
  { mark: 'snapshot', name: 'Each snapshot' },
  { mark: 'cell', name: 'Cell center' }
] as const

/**
 * Lays out the drawing's key, a column of its marks in a corner of the plot.
 */
const key = (): VNode =>
  h('ul', { 'aria-hidden': 'true', 'class': 'threat-key' }, KEY.map(({ mark, name }) =>
    h('li', [h('span', { class: `threat-key-${mark}` }), name])))

/**
 * Builds the rail's handlers, each moving `playback` to the time the reader
 * set, holding it under a press and letting it go on the release, then calling
 * `repaint`, since a drawing standing still paints only when told.
 */
const handlers = (playback: Playback, repaint: () => void) => ({
  onChange: (seconds: number) => {
    playback.seek(seconds)
    repaint()
  },

  onCommit: (seconds: number) => {
    playback.seek(seconds)
    playback.release()
    repaint()
  },

  onPointercancel: () => {
    playback.release()
    repaint()
  },

  onPointerdown: () => {
    playback.press()
    repaint()
  }
})

/**
 * The air along the ground as the threat's front passes, a drawing of the model
 * of the front under a rail a reader can move, and beneath it the time, the
 * front, and the air at the first and the last cell of the grid. A drag on the
 * rail sets the time, an arrow key steps it one snapshot, Home and End take it
 * to either end, and a press holds the run until it lets go.
 */
export default defineComponent({
  name: 'ThreatFront',
  setup() {
    const front    = new Front()
    const playback = new Playback(front)
    const aim      = ref(0)
    const first    = ref<HTMLElement>()
    const reach    = ref<HTMLElement>()
    const last     = ref<HTMLElement>()
    const time     = ref<HTMLElement>()
    const rail     = ref<ComponentPublicInstance>()
    const stops    = Array.from({ length: front.snapshots }, (_, snapshot) => snapshot * front.interval)
    let stage: Stage | undefined
    const host     = useStage(box => {
      const readout = {
        aim,
        first : first.value!,
        front : reach.value!,
        last  : last.value!,
        rail  : rail.value!.$el as HTMLElement,
        time  : time.value!
      }
      stage         = drawThreat(front, box, playback, readout)
      return stage
    })
    const rails    = handlers(playback, () => stage?.repaint())
    const named    = `The air temperature ${front.lowest} m up against the distance east, `
                   + 'a front of hot air advancing from the western edge'
    return () => h('figure', { class: 'threat-front' }, [
      h('div', { class: 'threat-plot' }, [
        h('div', { 'aria-label': named, 'class': 'threat-canvas', 'ref': host, 'role': 'img' }),
        key(),
        h(Slider, {
          class  : 'threat-rail kit-slider-over',
          label  : 'Time since the first snapshot',
          max    : front.span,
          min    : 0,
          ref    : rail,
          spoken : (seconds: number) => spoken(front, seconds),
          stops,
          value  : aim.value,
          ...rails
        })
      ]),
      h('div', { class: 'threat-chips' }, [
        h(Chip, { name: 'Time', shape: '120 s', value: time }),
        h(Chip, { name: 'Front', shape: '120 m', value: reach }),
        h(Chip, { name: 'First cell', shape: '320.0 K', value: first }),
        h(Chip, { name: 'Last cell', shape: '320.0 K', value: last })
      ])
    ])
  }
})
