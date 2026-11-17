// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import { Select } from '../../src/controls/select.tsx'

/**
 * The choices these cases offer.
 */
const CHOICES = ['every snapshot', 'the last two snapshots']

/**
 * Mounts the menu with the choice at `picked` shown.
 */
const menu = (picked: number) =>
  mount(Select, { attachTo: document.body, props: { choices: CHOICES, label: 'When', picked } })

enableAutoUnmount(afterEach)

describe('Select', () => {
  it('names its button from its label and stacks every choice in it, showing the one picked', () => {
    const widget  = menu(1)
    const trigger = widget.get('[role=combobox]')
    expect(document.querySelector(`[id="${trigger.attributes('aria-labelledby')!}"]`)?.textContent).toBe('When')
    expect(widget.findAll('.kit-select-words > span').map(choice => choice.text())).toEqual(CHOICES)
    expect(widget.get('.kit-select-words [data-picked]').text()).toBe('the last two snapshots')
  })

  it('opens its list on a press and reports the place of the choice picked from it', async () => {
    const widget = menu(0)
    await widget.get('[role=combobox]').trigger('click')
    expect(widget.get('[role=listbox]').attributes('hidden')).toBeUndefined()
    await widget.findAll('[role=option]')[1]!.trigger('click')
    expect(widget.emitted('change')).toEqual([[1]])
  })

  it('shows whatever pick its caller passes back', async () => {
    const widget = menu(0)
    await widget.setProps({ picked: 1 })
    expect(widget.get('.kit-select-words [data-picked]').text()).toBe('the last two snapshots')
  })
})
