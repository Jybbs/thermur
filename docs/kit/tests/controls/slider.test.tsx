// @vitest-environment happy-dom
import { enableAutoUnmount, mount }        from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import { Slider, stepped } from '../../src/controls/slider.tsx'

/**
 * The stops a slider steps between in these cases, one every fifteen.
 */
const STOPS = [0, 15, 30, 45, 60, 75, 90, 105, 120]

/**
 * Mounts a slider from 0 to 120 at `value`, given `stops` or none.
 */
function slider(value: number, stops: readonly number[] = []) {
  const widget = mount(Slider, {
    attachTo : document.body,
    props    : { label: 'Time', max: 120, min: 0, spoken: (at: number) => `${at} seconds`, stops, value }
  })
  ;(widget.get('[role=slider]').element as HTMLElement).focus()
  return widget
}

enableAutoUnmount(afterEach)

describe('a step between stops', () => {
  it.for([
    { direction: 1, from: 60, name: 'the next stop from one', to: 75 },
    { direction: -1, from: 60, name: 'the last stop from one', to: 45 },
    { direction: 1, from: 52, name: 'the next stop from between two', to: 60 },
    { direction: -1, from: 52, name: 'the last stop from between two', to: 45 },
    { direction: 1, from: 120, name: 'the end, and no further', to: 120 },
    { direction: -1, from: 0, name: 'the start, and no further', to: 0 },
    { direction: -1, from: 45 + 1e-9, name: 'the last stop from a hair past one', to: 30 },
    { direction: 1, from: 45 - 1e-9, name: 'the next stop from a hair short of one', to: 60 }
  ] as const)('lands on $name', ({ direction, from, to }) => {
    expect(stepped(STOPS, from, direction)).toBe(to)
  })
})

describe('Slider', () => {
  it('names its thumb for a screen reader, beside its range and the words for its value', () => {
    expect(slider(30).get('[role=slider]').attributes()).toMatchObject({
      'aria-label'     : 'Time',
      'aria-valuemax'  : '120',
      'aria-valuemin'  : '0',
      'aria-valuenow'  : '30',
      'aria-valuetext' : '30 seconds'
    })
  })

  it('shows whatever value its caller passes back', async () => {
    const widget = slider(30)
    await widget.setProps({ value: 90 })
    expect(widget.get('[role=slider]').attributes('aria-valuenow')).toBe('90')
  })
})

describe('Slider’s keys', () => {
  it.for([
    { key: 'ArrowRight', to: 31 },
    { key: 'ArrowLeft', to: 29 },
    { key: 'ArrowUp', to: 31 },
    { key: 'ArrowDown', to: 29 },
    { key: 'Home', to: 0 },
    { key: 'End', to: 120 }
  ])('takes $key to $to without stops, reporting the change and its end', async ({ key, to }) => {
    const widget = slider(30)
    await widget.get('[role=slider]').trigger('keydown', { key })
    expect(widget.emitted('change')).toEqual([[to]])
    expect(widget.emitted('commit')).toEqual([[to]])
  })

  it.for([
    { key: 'ArrowRight', to: 40 },
    { key: 'ArrowUp', to: 40 },
    { key: 'ArrowLeft', to: 20 },
    { key: 'ArrowDown', to: 20 }
  ])('takes $key under Shift to $to, ten steps at once along or across the track', async ({ key, to }) => {
    const widget = slider(30)
    await widget.get('[role=slider]').trigger('keydown', { key, shiftKey: true })
    expect(widget.emitted('change')).toEqual([[to]])
    expect(widget.emitted('commit')).toEqual([[to]])
  })

  it.for([
    { key: 'ArrowRight', to: 60 },
    { key: 'ArrowUp', to: 60 },
    { key: 'ArrowLeft', to: 45 },
    { key: 'ArrowDown', to: 45 }
  ])('takes $key to the stop at $to, through its end alone', ({ key, to }) => {
    const widget  = slider(52, STOPS)
    const pressed = new KeyboardEvent('keydown', { cancelable: true, key })
    widget.get('[role=slider]').element.dispatchEvent(pressed)
    expect(pressed.defaultPrevented).toBe(true)
    expect(widget.emitted('commit')).toEqual([[to]])
    expect(widget.emitted('change')).toBeUndefined()
  })

  it('leaves Home and End to the machine where it steps between stops', async () => {
    const widget = slider(52, STOPS)
    await widget.get('[role=slider]').trigger('keydown', { key: 'End' })
    expect(widget.emitted('commit')).toEqual([[120]])
  })
})
