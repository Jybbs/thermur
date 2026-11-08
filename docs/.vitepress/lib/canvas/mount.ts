import { type Ref, onBeforeUnmount, onMounted, ref } from 'vue'

import type { Stage } from './stage.ts'

/**
 * Builds the box a component's drawing fills, which `draw` fills once the
 * component mounts and whose drawing is destroyed once it unmounts, since the
 * site moves between pages without a reload.
 */
export function useStage(draw: (host: HTMLElement) => Pick<Stage, 'destroy'>): Ref<HTMLElement | undefined> {
  const host  = ref<HTMLElement>()
  let drawing : Pick<Stage, 'destroy'> | undefined
  onMounted(() => {
    drawing = draw(host.value!)
  })
  onBeforeUnmount(() => drawing?.destroy())
  return host
}
