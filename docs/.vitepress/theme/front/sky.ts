import { defineComponent, h } from 'vue'

import { useStage }  from '../../lib/canvas/mount.ts'
import { drawFront } from '../../lib/front/drawing.ts'

/**
 * The front page's background, a box pinned behind the whole page that the
 * drawing fills once the page mounts and empties once it unmounts.
 */
export default defineComponent({
  name: 'FrontSky',
  setup() {
    const host = useStage(drawFront)
    return () => h('div', { 'aria-hidden': 'true', 'class': 'front-sky', 'ref': host })
  }
})
