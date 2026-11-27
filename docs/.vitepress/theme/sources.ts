import { type VNode, defineComponent, h, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'

import { tracedGlyphs }      from '../lib/canvas/marks.ts'
import source                from '../lib/rendered/identity.json' with { type: 'json' }
import { tenths }            from '../lib/scene.ts'
import { type Bird, fillOf } from '../lib/sources/drawing.ts'
import { watchSources }      from '../lib/sources/page.ts'

/**
 * The path data of each traced bird of the mark.
 */
const birdPaths = tracedGlyphs(source)

/**
 * Builds the flock's nodes: the traced birds once as paths and a use of one
 * per bird of `birds`, or no node where `birds` is empty.
 */
const nodesOf = (birds: Bird[]): VNode[] => birds.length === 0 ? [] : [
  h('defs', birdPaths.map((d, glyph) => h('path', { d, id: `sources-glyph-${glyph}` }))),
  ...birds.map(bird => h('use', {
    href      : `#sources-glyph-${bird.glyph}`,
    style     : { fill: fillOf(bird) },
    transform : placed(bird)
  }))
]

/**
 * Writes where a bird sits as an SVG transform, its head at the origin of its
 * traced bird moved to the bird's place, turned, and scaled to its length.
 */
const placed = ({ angle, length, x, y }: Bird): string =>
  `translate(${tenths(x)} ${tenths(y)}) rotate(${tenths(angle * 180 / Math.PI)}) scale(${tenths(length)})`

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
      const section = host.value!.closest<HTMLElement>('.sources')!
      stop          = watchSources(section, birdPaths.length, laid => birds.value = laid)
    })
    onBeforeUnmount(() => stop?.())
    const props = { 'aria-hidden': 'true', 'class': 'sources-flock', 'ref': host }
    return () => h('svg', props, nodesOf(birds.value))
  }
})
