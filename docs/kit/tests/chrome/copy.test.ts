// @vitest-environment happy-dom
import { mount }                     from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h, watch } from 'vue'

import * as copy from '../../src/chrome/copy.ts'

/**
 * A component holding a copier, a button that copies `text` through it, and
 * the words the copy says.
 */
function mounted(text: string): { copier: copy.Copier, unmount: () => void, button: HTMLButtonElement } {
  let copier: copy.Copier | undefined
  const wrapper = mount(defineComponent({
    setup() {
      copier = copy.useCopy()
      return () => h('button', { onClick: (event: Event) => copier!.copy(text, event) })
    }
  }), { attachTo: document.body })
  return { button: wrapper.element as HTMLButtonElement, copier: copier!, unmount: () => wrapper.unmount() }
}

/**
 * Presses `button` and waits for the copy to land.
 */
async function press(button: HTMLButtonElement): Promise<void> {
  button.click()
  await vi.waitFor(() => expect(button.dataset.copied).toBe(''))
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
})

afterEach(() => {
  vi.useRealTimers()
  document.body.replaceChildren()
})

describe('useCopy', () => {
  it('copies a hex, marks the control in it for the hold, and says what it copied', async () => {
    const { button, copier } = mounted('#c38243')
    await press(button)
    expect(await navigator.clipboard.readText()).toBe('#c38243')
    expect(button.style.getPropertyValue('--copied')).toBe('#c38243')
    expect(copier.said.value).toBe('Copied #c38243')
    vi.advanceTimersByTime(copy.COPY_HOLD)
    expect(button.dataset.copied).toBeUndefined()
  })

  it('draws a copy that is no hex in the lead, through no color of its own', async () => {
    const { button } = mounted("'Bellefair', Didot, Georgia, serif")
    button.style.setProperty('--copied', '#000000')
    await press(button)
    expect(button.style.getPropertyValue('--copied')).toBe('')
  })

  it('starts the hold over on a second press, ending it a full hold after that press', async () => {
    const { button, copier } = mounted('#5069ad')
    const pressed            = { currentTarget: button } as unknown as Event
    await copier.copy('#5069ad', pressed)
    vi.advanceTimersByTime(copy.COPY_HOLD - 100)
    await copier.copy('#5069ad', pressed)
    vi.advanceTimersByTime(copy.COPY_HOLD - 1)
    expect(button.dataset.copied).toBe('')
    vi.advanceTimersByTime(1)
    expect(button.dataset.copied).toBeUndefined()
  })

  it('ends every running hold as the widget unmounts', async () => {
    const { button, unmount } = mounted('#5069ad')
    await press(button)
    const cleared = vi.spyOn(globalThis, 'clearTimeout')
    unmount()
    expect(cleared).toHaveBeenCalledOnce()
  })
})

describe('what useCopy says', () => {
  it('empties what it says before each copy, so a second copy of one value is heard again', async () => {
    const { button, copier } = mounted('#5069ad')
    const pressed            = { currentTarget: button } as unknown as Event
    const heard: string[]    = []
    watch(copier.said, words => heard.push(words), { flush: 'sync' })
    await copier.copy('#5069ad', pressed)
    await copier.copy('#5069ad', pressed)
    expect(heard).toEqual(['Copied #5069ad', '', 'Copied #5069ad'])
  })

  it('says a copy the clipboard refuses failed, and leaves the control unmarked', async () => {
    vi.spyOn(navigator.clipboard, 'writeText').mockRejectedValueOnce(new Error('denied'))
    const { button, copier } = mounted('#5069ad')
    button.click()
    await vi.waitFor(() => expect(copier.said.value).toBe('Could not copy #5069ad'))
    expect(button.dataset.copied).toBeUndefined()
  })
})
