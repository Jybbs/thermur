import { type Ref, onBeforeUnmount, onMounted, ref } from 'vue'

import type { Stage } from './stage.ts'

/**
 * Builds the box a component's drawing fills, which `draw` fills once the
 * component mounts and which empties once it unmounts, since the site moves
 * between pages without a reload.
 */
export function useStage(draw: (host: HTMLElement) => Stage): Ref<HTMLElement | undefined> {
  const host = ref<HTMLElement>()
  let stage  : Stage | undefined
  onMounted(() => {
    stage = draw(host.value!)
  })
  onBeforeUnmount(() => stage?.destroy())
  return host
}
