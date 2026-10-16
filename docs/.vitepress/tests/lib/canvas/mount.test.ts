// @vitest-environment happy-dom
import { mount }              from '@vue/test-utils'
import { expect, it, vi }     from 'vitest'
import { defineComponent, h } from 'vue'

import { useStage }   from '../../../lib/canvas/mount.ts'
import type { Stage } from '../../../lib/canvas/stage.ts'

it('draws into the component’s box once it mounts and destroys the stage once it unmounts', () => {
  const stage = { destroy: vi.fn<() => void>() }
  const draw  = vi.fn<(host: HTMLElement) => Stage>(() => stage as unknown as Stage)
  const box   = mount(defineComponent({
    setup: () => {
      const host = useStage(draw)
      return () => h('div', { ref: host })
    }
  }))
  expect(draw).toHaveBeenCalledExactlyOnceWith(box.element)
  expect(stage.destroy).not.toHaveBeenCalled()
  box.unmount()
  expect(stage.destroy).toHaveBeenCalledOnce()
})
