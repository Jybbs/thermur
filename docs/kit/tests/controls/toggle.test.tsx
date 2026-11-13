// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import { Toggle } from '../../src/controls/toggle.tsx'

enableAutoUnmount(afterEach)

describe('Toggle', () => {
  it('reports each press and shows the state its caller passes back, its slot drawing the face', async () => {
    const widget = mount(Toggle, {
      props : { label: 'Pause', pressed: false },
      slots : { default: ({ pressed }: { pressed: boolean }) => String(pressed) }
    })
    const button = widget.get('button')
    expect([button.attributes('aria-label'), button.attributes('aria-pressed'), button.text()])
      .toEqual(['Pause', 'false', 'false'])
    await button.trigger('click')
    expect(widget.emitted('change')).toEqual([[true]])
    await widget.setProps({ pressed: true })
    expect([button.attributes('aria-pressed'), button.text()]).toEqual(['true', 'true'])
  })
})
