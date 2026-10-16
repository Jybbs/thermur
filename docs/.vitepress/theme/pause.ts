import { useData }                            from 'vitepress'
import { defineComponent, h, onMounted, ref } from 'vue'

import { pause, paused } from '../lib/canvas/pause.ts'
import { animated }      from '../lib/page-data.ts'

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
    const toggle = (): void => {
      pressed.value = !pressed.value
      pause(pressed.value)
    }
    return () => animated(page.value)
      ? h('button', {
          'aria-label'   : 'Pause the motion on this page',
          'aria-pressed' : String(pressed.value),
          'class'        : 'motion-pause',
          'onClick'      : toggle,
          'type'         : 'button'
        }, h('span', { 'aria-hidden': 'true', 'class': `motion-pause-${pressed.value ? 'play' : 'bars'}` }))
      : null
  }
})
