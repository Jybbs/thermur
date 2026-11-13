// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { defineComponent, h, ref }         from 'vue'

import { Chip } from '../../src/chrome/chip.tsx'

enableAutoUnmount(afterEach)

describe('Chip', () => {
  it('sets its name and hands its caller the boxes its drawing writes into, as wide as its shape', () => {
    const named  = ref<HTMLElement>()
    const value  = ref<HTMLElement>()
    const widget = mount(defineComponent({ setup: () => () => h(Chip, { name: 'Time', named, shape: '120 s', value }) }))
    value.value!.textContent = '60 s'
    expect(widget.text()).toBe('Time60 s')
    expect(named.value!.textContent).toBe('Time')
    expect(value.value!.style.minWidth).toBe('5ch')
  })
})
