import { Chip, Slider }                      from '@thermur/kit'
import { type Ref, defineComponent, h, ref } from 'vue'

import { useStage }               from '../lib/canvas/mount.ts'
import type { Readout }           from '../lib/lookout/drawing.ts'
import { DRONES, played, spoken } from '../lib/lookout/flocks.ts'
import { LookoutStage }           from '../lib/lookout/stage.ts'

/**
 * The boxes each reading of a `Readout` is written into, before the page
 * mounts them.
 */
type Boxes = Record<keyof Readout, Ref<HTMLElement | undefined>>

/**
 * The share of the slider's track the plan's default size sits at.
 */
const SHARE = `${100 * (DRONES.value - DRONES.min) / (DRONES.max - DRONES.min)}%`

/**
 * Draws the header holding the slider that sets the flock's size at `size`,
 * its default marked on the track, flying the size let go through `fly`.
 */
const bar = (size: Ref<number>, fly: (count: number) => Promise<void>) =>
  h('div', { class: 'lookout-bar' }, [
    h('span', { class: 'lookout-size' }, [
      h('span', { 'aria-hidden': 'true', 'class': 'kit-slider-label' }, 'Drones'),
      h('span', { class: 'lookout-track' }, [
        h(Slider, {
          label    : 'Drones in the flock',
          max      : DRONES.max,
          min      : DRONES.min,
          spoken,
          step     : DRONES.step,
          stops    : DRONES.stops,
          value    : size.value,
          onChange : (count: number) => {
            size.value = count
          },
          onCommit : fly
        }),
        h('span', { 'aria-hidden': 'true', 'class': 'lookout-default', 'style': { left: SHARE } })
      ])
    ])
  ])

/**
 * Draws the chips the Lookout reads the drone count, the time flown, the
 * polarization, and the number of groups into.
 */
const chips = (boxes: Boxes) =>
  h('div', { class: 'lookout-chips' }, [
    h(Chip, { name: 'Drones', shape: '3,000', value: boxes.drones }),
    h(Chip, { name: 'Time', shape: '120.0 s', value: boxes.time }),
    h(Chip, { name: 'Polarization', shape: '0.00', value: boxes.polarization }),
    h(Chip, { name: 'Groups', shape: '000', value: boxes.groups })
  ])

/**
 * Reads the mounted boxes as the readout the Lookout writes into.
 */
const readout = (boxes: Boxes): Readout => ({
  drones       : boxes.drones.value!,
  groups       : boxes.groups.value!,
  polarization : boxes.polarization.value!,
  time         : boxes.time.value!
})

/**
 * The Lookout: a flight of the composite seen from the side, beneath a header
 * holding the slider that sets the flock's size, its default marked on the
 * track, and beside the flight the drone count, the time flown, the
 * polarization, and the number of groups, read as the flight plays. A size let
 * go flies the flight again from the same seed, in the page, or through the
 * package's graph where `thermur watch` opened the page, the size let go last
 * playing however the flights before it finish.
 */
export default defineComponent({
  name: 'LookoutFlight',
  setup() {
    const size  = ref<number>(DRONES.value)
    const boxes = { drones: ref<HTMLElement>(), groups: ref<HTMLElement>(), polarization: ref<HTMLElement>(), time: ref<HTMLElement>() }
    let asked   = 0
    let drawing: LookoutStage | undefined
    let served: string | null = null
    const host  = useStage(box => {
      let gone = false
      served   = new URLSearchParams(window.location.search).get('flight')
      const start = async (): Promise<void> => {
        const playing = await played(DRONES.value, served)
        if (!gone) drawing = new LookoutStage(box, playing, readout(boxes))
      }
      void start()
      return {
        destroy: () => {
          gone = true
          drawing?.destroy()
        }
      }
    })
    const fly   = async (count: number): Promise<void> => {
      const flight  = ++asked
      size.value    = count
      const playing = await played(count, served)
      if (flight === asked) drawing?.play(playing)
    }
    return () => h('figure', { class: 'lookout' }, [
      bar(size, fly),
      h('div', { class: 'lookout-flight' }, [h('div', { class: 'lookout-stage', ref: host }), chips(boxes)])
    ])
  }
})
