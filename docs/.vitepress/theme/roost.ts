import { Chip, still } from '@thermur/kit'
import { type Ref, type VNode, defineComponent, h, onBeforeUnmount, onMounted, ref } from 'vue'

import { Stage }           from '../lib/canvas/stage.ts'
import { flight }          from '../lib/roost/flight.ts'
import { Player }          from '../lib/roost/player.ts'
import { partsOf }         from '../lib/roost/parts.ts'
import { noteOf }          from '../lib/roost/shot.ts'
import { READINGS }        from '../lib/roost/strip.ts'

/**
 * What the view writes as the moment shown moves: the clock, the ringed
 * drone's height, and each reading by its index.
 */
type Said = 'clock' | 'led' | `reading-${number}`

/**
 * The longest each value is written, which its chip stands as wide as.
 */
const SHAPES: Partial<Record<Said, string>> = { 'clock': '300 s', 'led': '200 m' }

/**
 * Lays out one reading: its name in capitals beside its value, its chart, and
 * what it is held against.
 */
const reading = (index: number, value: VNode): VNode => {
  const { against, band, legend, name } = READINGS[index]!
  return h('div', { class: 'roost-reading' }, [
    h('span', { class: 'roost-caps roost-name' }, name),
    value,
    h('div', { 'aria-hidden': 'true', 'class': 'roost-strip', 'data-box': `strip-${index}` }),
    h('span', { 'class': 'roost-against roost-caps', 'data-kind': band[0] === band[1] ? 'line' : 'band' }, legend ?? against)
  ])
}

/**
 * Lays out the view: the scene with the clock and the ringed drone's height
 * over it, beside the three readings and the note, each value in the box
 * `slot` gives its key.
 */
const frame = (root: Ref<HTMLElement | undefined>, slot: (key: Said) => Ref<HTMLElement | undefined>): VNode => {
  const chip = (key: Said, name: string): VNode => h(Chip, { name, shape: SHAPES[key] ?? '', value: slot(key) })
  return h('figure', { class: 'roost', ref: root }, [
    h('div', { class: 'roost-frame' }, [
      h('div', { class: 'roost-stage' }, [
        h('div', {
          'aria-label' : 'Two hundred drones flying over the roost around a fire spreading from its middle, the view moving with the ringed one',
          'class'      : 'roost-scene',
          'data-box'   : 'scene',
          'role'       : 'img'
        }),
        h('div', { class: 'roost-over' }, [chip('clock', 'Time'), chip('led', 'Ringed height')])
      ]),
      h('div', { class: 'roost-side' }, [
        ...READINGS.map((_, index) => reading(index, h('span', { class: 'roost-value', ref: slot(`reading-${index}`) }, ''))),
        h('p', { class: 'roost-note' }, noteOf(flight()))
      ])
    ])
  ])
}

/**
 * Two hundred drones flying over the roost around a fire spreading from its
 * middle, the view moving with a ringed one and panning in from the whole
 * roost as the flight starts. The fire is drawn as burned air in bands of the
 * heat ramp, each drone reddening the harder its heat pushes it, and beside the
 * flock sit the clock, the ringed drone's height, and the polarization, the
 * spacing, and the height of the whole flock over the flight. Every drone
 * flies the site's parity-tested copy of the package's composite, flown once
 * however long the page stays open.
 */
export default defineComponent({
  name: 'RoostView',
  setup() {
    const root    = ref<HTMLElement>()
    const said    = new Map<Said, Ref<HTMLElement | undefined>>()
    const stages  : Stage[] = []
    const slot    = (key: Said): Ref<HTMLElement | undefined> => {
      if (!said.has(key)) said.set(key, ref<HTMLElement>())
      return said.get(key)!
    }
    let unmounted = false
    onMounted(async () => {
      if (still()) await flight().ready()
      if (unmounted) return
      const host   = root.value!
      const player = new Player(flight())
      const at     = new Float32Array(3)
      const words  = (key: Said): string => {
        if (key === 'clock') return `${Math.floor(player.t)} s`
        if (key === 'led') {
          player.place(player.flight.led, at)
          return `${at[2]!.toFixed(0)} m`
        }
        const { of, write } = READINGS[Number(key.slice(8))]!
        return write(player.read(of(player)))
      }
      const shown  = (): void => {
        for (const [key, box] of said) {
          const written = words(key)
          if (box.value && box.value.textContent !== written) box.value.textContent = written
        }
        host.style.setProperty('--veil', player.veil.toFixed(3))
      }
      for (const [name, part] of Object.entries(partsOf(player, shown))) {
        stages.push(new Stage(host.querySelector<HTMLElement>(`[data-box="${name}"]`)!, [part], { pinned: false }))
      }
    })
    onBeforeUnmount(() => {
      unmounted = true
      for (const stage of stages) stage.destroy()
    })
    return () => frame(root, slot)
  }
})
