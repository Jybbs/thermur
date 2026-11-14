import { Chip, Radios, Select, Slider, idle, tween, useLoop }                               from '@thermur/kit'
import { type Component, defineComponent, h, onBeforeUnmount, onMounted, ref, watchEffect } from 'vue'

import { FUELS, GROUND, SPAN, WINDS } from '../../lib/kit/front.ts'
import { playing, turn, useSteps }    from './parts.ts'
import { Gate, Width }                from './scenes.ts'

/**
 * The readings the chip's example steps through, the longest setting its
 * width.
 */
const READINGS = ['0 m', '18 m', '72 m', '144 m', '300 m'] as const

/**
 * A bar easing from empty to full over three times the theme's settling time
 * on each press, starting again from empty on a second one.
 */
const Ease = defineComponent({
  name: 'KitEase',
  setup() {
    const share = ref(1)
    let stop    = idle
    onBeforeUnmount(() => stop())
    return () => h('div', { class: 'kit-catalog-row' }, [
      turn('Ease again', () => {
        stop()
        share.value = 0
        stop        = tween(now => share.value = now, 3)
      }),
      h('span', { class: 'kit-catalog-bar', style: { '--eased': share.value } })
    ])
  }
})

/**
 * The example of each piece, by the piece's key and then the example's name
 * its declaration gives it, each holding state of its own apart from the
 * stage's.
 */
export const EXAMPLES: Readonly<Record<string, Readonly<Record<string, Component>>>> = {
  chip: {
    'A Reading Over a Front': defineComponent({
      setup() {
        const place = ref(2)
        const value = ref<HTMLElement>()
        onMounted(() => watchEffect(() => value.value!.textContent = READINGS[place.value]!))
        return () => h('div', { class: 'kit-catalog-row' }, [
          h(Chip, { name: 'Reach', shape: `${GROUND} m`, value }),
          turn('Step', () => place.value = (place.value + 1) % READINGS.length)
        ])
      }
    })
  },
  copy: {
    'The Heat Ramp': defineComponent(useSteps)
  },
  gate: {
    'The Page\'s Pause': Gate
  },
  loop: {
    'A Running Clock': defineComponent({
      setup() {
        const seconds = ref(0)
        const loop    = useLoop(step => seconds.value += step)
        return () => h('div', { class: 'kit-catalog-row' }, [
          playing('Run the clock', loop.running.value, loop.run),
          h('span', { class: 'kit-catalog-said' }, `${seconds.value.toFixed(1)} s`)
        ])
      }
    })
  },
  markup: {
    'Drawn at Each Width': Width
  },
  radios: {
    'Three Winds': defineComponent({
      setup() {
        const wind = ref<string>(WINDS[1].value)
        return () => h(Radios, { label: 'Wind', options: WINDS, value: wind.value, onChange: (value: string) => wind.value = value })
      }
    })
  },
  select: {
    'Inside a Sentence': defineComponent({
      setup() {
        const fuel = ref(1)
        return () => h('p', { class: 'kit-catalog-sentence' }, [
          'Burn ',
          h(Select, { choices: FUELS.map(one => one.words), label: 'Fuel', picked: fuel.value, onChange: (place: number) => fuel.value = place }),
          ' under the breeze.'
        ])
      }
    })
  },
  slider: {
    'Between Stops': defineComponent({
      setup() {
        const seconds = ref(30)
        const moved   = (value: number): void => {
          seconds.value = value
        }
        return () => h('div', { class: 'kit-catalog-row kit-catalog-wide' }, [
          h(Slider, {
            label    : 'Seconds since ignition',
            max      : SPAN,
            min      : 0,
            spoken   : (at: number) => `${Math.round(at)} seconds`,
            stops    : [0, 1, 2, 3, 4].map(quarter => quarter * SPAN / 4),
            value    : seconds.value,
            onChange : moved,
            onCommit : moved
          }),
          h('span', { class: 'kit-catalog-said' }, `${Math.round(seconds.value)} s`)
        ])
      }
    })
  },
  timing: {
    'Ease Again': Ease
  },
  toggle: {
    'Play and Hold': defineComponent({
      setup() {
        const on = ref(false)
        return () => h('div', { class: 'kit-catalog-row' }, [
          playing('Play', on.value, (now: boolean) => on.value = now),
          h('span', { class: 'kit-catalog-said' }, on.value ? 'Pressed' : 'Not pressed')
        ])
      }
    })
  }
}
