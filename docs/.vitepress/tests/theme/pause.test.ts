// @vitest-environment happy-dom
import { mount }                               from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ref }                                 from 'vue'

import MotionPause from '../../theme/pause.ts'

const page = ref<{ isNotFound?: boolean, relativePath: string }>({ relativePath: 'index.md' })

vi.mock('vitepress', () => ({ useData: () => ({ page }) }))

afterEach(() => {
  delete document.documentElement.dataset.paused
  page.value = { relativePath: 'index.md' }
})

describe('MotionPause', () => {
  it('pauses and resumes the page’s motion through the root element', async () => {
    const root   = document.documentElement
    const pause  = mount(MotionPause)
    const button = pause.get('button')
    expect(button.attributes('aria-pressed')).toBe('false')
    await button.trigger('click')
    expect([button.attributes('aria-pressed'), root.dataset.paused]).toEqual(['true', 'true'])
    await button.trigger('click')
    expect([button.attributes('aria-pressed'), root.dataset.paused]).toEqual(['false', 'false'])
  })

  it('shows as paused once mounted on a page whose motion is already paused', async () => {
    document.documentElement.dataset.paused = 'true'
    const pause                             = mount(MotionPause)
    await pause.vm.$nextTick()
    expect(pause.get('button').attributes('aria-pressed')).toBe('true')
    expect(pause.find('.motion-pause-play').exists()).toBe(true)
  })

  it.for([
    [{ isNotFound: true, relativePath: '404.md' }, true],
    [{ relativePath: 'methods/alarm.md' }, false]
  ] as const)('shows on %j only where its drawing moves on its own: %s', ([shown, expected]) => {
    page.value = shown
    expect(mount(MotionPause).find('button').exists()).toBe(expected)
  })
})
