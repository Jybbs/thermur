import { Toggle, pause, paused }              from '@thermur/kit'
import { useData }                            from 'vitepress'
import { defineComponent, h, onMounted, ref } from 'vue'

import { animated } from '../lib/moving.ts'

/**
 * The switch that pauses a page's motion, shown in the navigation bar on each
 * page carrying a drawing that moves on its own. It sets `data-paused` on the
 * root element, which stands every drawing still, and shows as playing until it
 * mounts, so the page the server rendered matches the first one the browser
 * draws.
 */
export default defineComponent({
  name: 'MotionPause',
  setup() {
    const { page } = useData()
    const pressed  = ref(false)
    onMounted(() => {
      pressed.value = paused()
    })
    const pressedTo = (on: boolean): void => {
      pressed.value = on
      pause(on)
    }
    return () => animated(page.value)
      ? h(Toggle, {
          class    : 'motion-pause',
          label    : 'Pause the motion on this page',
          onChange : pressedTo,
          pressed  : pressed.value
        }, {
          default: ({ pressed: on }: { pressed: boolean }) =>
            h('span', { 'aria-hidden': 'true', 'class': `motion-pause-${on ? 'play' : 'bars'}` })
        })
      : null
  }
})
