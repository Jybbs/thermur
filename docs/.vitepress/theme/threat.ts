import { type Ref, type VNode, defineComponent, h, ref } from 'vue'

import { useStage }   from '../lib/canvas/mount.ts'
import type { Stage } from '../lib/canvas/stage.ts'
import { drawThreat } from '../lib/threat/drawing.ts'
import { Front }      from '../lib/threat/front.ts'
import { Playback }   from '../lib/threat/playback.ts'

/**
 * What each arrow key does to the time on the rail.
 */
const ARROWS: Readonly<Record<string, -1 | 1>> = { ArrowDown: -1, ArrowLeft: -1, ArrowRight: 1, ArrowUp: 1 }
/**
 * The marks the drawing's key names, by the class giving each its form.
 */
const KEY = [
  { mark: 'now', name: 'Air now' },
  { mark: 'snapshot', name: 'Each snapshot' },
  { mark: 'cell', name: 'Cell center' }
] as const

/**
 * Lays out one chip beneath the drawing: its name in capitals and its value, a
 * box the drawing writes into, as wide as `shape` so the row never shifts as
 * the value changes.
 */
const chip = (label: string, shape: string, value: Ref<HTMLElement | undefined>): VNode =>
  h('span', { class: 'threat-chip' }, [
    h('span', { class: 'threat-chip-name' }, label),
    h('span', { class: 'threat-chip-value', ref: value, style: { minWidth: `${shape.length}ch` } })
  ])

/**
 * Lays out the drawing's key, a column of its marks in a corner of the plot.
 */
const key = (): VNode =>
  h('ul', { 'aria-hidden': 'true', 'class': 'threat-key' }, KEY.map(({ mark, name }) =>
    h('li', [h('span', { class: `threat-key-${mark}` }), name])))

/**
 * Builds what the rail does: a move sets the time, an arrow key steps it one
 * snapshot and Home and End take it to either end, and a press holds the run
 * until it lets go, each followed by `moved`, since a drawing standing still
 * paints only when told.
 *
 * Args:
 *   front    : The model of the threat, whose span the rail runs across.
 *   moved    : What runs after each move.
 *   playback : The clock the rail moves.
 */
function railHandlers(
  front    : Front,
  moved    : () => void,
  playback : Playback
): Record<string, (event: never) => void> {
  const release = (): void => {
    playback.release()
    moved()
  }
  return {
    onPointercancel : release,
    onPointerup     : release,

    onInput: (event: Event) => {
      playback.seek((event.target as HTMLInputElement).valueAsNumber)
      moved()
    },

    onKeydown: (event: KeyboardEvent) => {
      const direction = ARROWS[event.key]
      const edge      = { End: front.span, Home: 0 }[event.key]
      if (direction === undefined && edge === undefined) return
      event.preventDefault()
      if (direction === undefined) playback.seek(edge!)
      else playback.nudge(direction)
      moved()
    },

    onPointerdown: () => {
      playback.press()
      moved()
    }
  }
}

/**
 * The air along the ground as the threat's front passes, a drawing of the model
 * of the front over a rail a reader can move, and beneath it the time, the
 * front, and the air at the first and the last cell of the grid.
 */
export default defineComponent({
  name: 'ThreatFront',
  setup() {
    const front    = new Front()
    const playback = new Playback(front)
    const first    = ref<HTMLElement>()
    const reach    = ref<HTMLElement>()
    const last     = ref<HTMLElement>()
    const time     = ref<HTMLElement>()
    const rail     = ref<HTMLInputElement>()
    let stage: Stage | undefined
    const host     = useStage(box => {
      const readout = {
        first : first.value!,
        front : reach.value!,
        last  : last.value!,
        rail  : rail.value!,
        time  : time.value!
      }
      stage         = drawThreat(front, box, playback, readout)
      return stage
    })
    const handlers = railHandlers(front, () => stage?.repaint(), playback)
    const named    = `The air temperature ${front.lowest} m up against the distance east, `
                   + 'a front of hot air advancing from the western edge'
    return () => h('figure', { class: 'threat-front' }, [
      h('div', { class: 'threat-plot' }, [
        h('div', { 'aria-label': named, 'class': 'threat-canvas', 'ref': host, 'role': 'img' }),
        key(),
        h('input', {
          ...handlers,
          'aria-label' : 'Time since the first snapshot',
          'class'      : 'threat-rail',
          'max'        : front.span,
          'min'        : 0,
          'ref'        : rail,
          'step'       : 1,
          'type'       : 'range',
          'value'      : 0
        })
      ]),
      h('div', { class: 'threat-chips' }, [
        chip('Time', '120 s', time),
        chip('Front', '120 m', reach),
        chip('First cell', '320.0 K', first),
        chip('Last cell', '320.0 K', last)
      ])
    ])
  }
})
