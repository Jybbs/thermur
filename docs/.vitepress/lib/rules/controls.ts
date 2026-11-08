/**
 * What each arrow key does to the choice in a group of radios.
 */
const ARROWS: Readonly<Record<string, -1 | 1>> = { ArrowDown: 1, ArrowLeft: -1, ArrowRight: 1, ArrowUp: -1 }

/**
 * Makes one chip of a readout, its name in capitals beside a value the drawing
 * writes into, and returns it with the name's box and the value's.
 */
export function chip(name: string): { element: HTMLElement, key: HTMLElement, value: HTMLElement } {
  const element     = document.createElement('span')
  const key         = document.createElement('span')
  const value       = document.createElement('span')
  element.className = 'rules-chip'
  key.className     = 'rules-chip-name'
  value.className   = 'rules-chip-value'
  key.textContent   = name
  element.append(key, value)
  return { element, key, value }
}

/**
 * Makes a group of radios as the ARIA radio-group pattern sets, a press picking
 * one and the arrow keys moving the choice around the group, each choice
 * passed to `picked`.
 *
 * Args:
 *   initial : The value checked at first.
 *   label   : The group's name, for a screen reader.
 *   options : Each radio's value beside its words.
 *   picked  : What runs with each value picked.
 */
export function radios<Value extends string>(
  initial : Value,
  label   : string,
  options : readonly (readonly [Value, string])[],
  picked  : (value: Value) => void
): HTMLElement {
  const group   = document.createElement('div')
  const buttons = options.map(([value, words]) => {
    const button         = document.createElement('button')
    button.type          = 'button'
    button.dataset.value = value
    button.textContent   = words
    button.setAttribute('role', 'radio')
    return button
  })
  const mark    = (value: Value, focus: boolean): void => {
    for (const button of buttons) {
      const on        = button.dataset.value === value
      button.tabIndex = on ? 0 : -1
      button.setAttribute('aria-checked', String(on))
      if (on && focus) button.focus()
    }
  }
  const choose  = (value: Value, focus: boolean): void => {
    mark(value, focus)
    picked(value)
  }
  buttons.forEach((button, at) => {
    button.addEventListener('click', () => choose(options[at]![0], false))
    button.addEventListener('keydown', event => {
      const step = ARROWS[event.key]
      if (step === undefined) return
      event.preventDefault()
      choose(options[(at + step + options.length) % options.length]![0], true)
    })
  })
  mark(initial, false)
  group.className = 'rules-radios'
  group.setAttribute('role', 'radiogroup')
  group.setAttribute('aria-label', label)
  group.append(...buttons)
  return group
}

/**
 * Makes a native slider, whose keyboard and pointer handling the ARIA slider
 * pattern accepts, its caption `words` beside it and its name for a screen
 * reader `named`, calling `moved` with each value it takes.
 */
export function slider(
  named : string,
  range : { max: number, min: number, step: number, value: number },
  words : string,
  moved : (value: number) => void
): HTMLElement {
  const input     = document.createElement('input')
  const label     = document.createElement('label')
  input.type      = 'range'
  input.max       = String(range.max)
  input.min       = String(range.min)
  input.step      = String(range.step)
  input.value     = String(range.value)
  label.className = 'rules-range'
  input.setAttribute('aria-label', named)
  input.addEventListener('input', () => moved(input.valueAsNumber))
  label.append(words, input)
  return label
}
