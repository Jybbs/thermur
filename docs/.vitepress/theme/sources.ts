import { defineComponent, h, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'

import { tracedGlyphs }                    from '../lib/canvas/marks.ts'
import source                              from '../lib/rendered/identity.json' with { type: 'json' }
import { type Bird, fillOf, watchSources } from '../lib/sources.ts'

/**
 * The path data of each traced bird of the mark.
 */
const GLYPHS = tracedGlyphs(source)

/**
 * Writes where a bird sits as an SVG transform, its head at the origin of its
 * traced bird moved to the bird's place, turned, and scaled to its length.
 */
const placed = ({ angle, length, x, y }: Bird): string =>
  `translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${(angle * 180 / Math.PI).toFixed(1)}) scale(${length.toFixed(2)})`

/**
 * The flock of the mark's birds over a page's Sources list, each number the
 * label of one bird, drawn once the page is in a browser and again whenever the
 * list changes width or another source becomes current. The list reads whole
 * without it, and it catches no pointer.
 */
export default defineComponent({
  name: 'SourcesFlock',
  setup() {
    const host  = ref<SVGSVGElement>()
    const birds = shallowRef<Bird[]>([])
    let stop: (() => void) | undefined
    onMounted(() => {
      stop = watchSources(host.value!.closest<HTMLElement>('.sources')!, GLYPHS.length, laid => birds.value = laid)
    })
    onBeforeUnmount(() => stop?.())
    return () => h('svg', { 'aria-hidden': 'true', 'class': 'sources-flock', 'ref': host }, birds.value.length === 0 ? [] : [
      h('defs', GLYPHS.map((d, glyph) => h('path', { d, id: `sources-glyph-${glyph}` }))),
      ...birds.value.map(bird => h('use', {
        href      : `#sources-glyph-${bird.glyph}`,
        style     : { fill: fillOf(bird) },
        transform : placed(bird)
      }))
    ])
  }
})
