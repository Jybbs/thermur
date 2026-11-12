import { type Ref, type VNode, h } from 'vue'

/**
 * Lays out the words a screen reader hears once a copy lands, out of sight.
 */
export const heard = (said: Ref<string>): VNode =>
  h('span', { 'aria-live': 'polite', 'class': 'visually-hidden' }, said.value)

/**
 * Lays out a key: a swatch painted `swatch`, `label` in capitals, and the hex
 * `hex`, a press on which runs `pressed`.
 */
export const keyOf = (label: string, swatch: string, hex: VNode, pressed: (event: Event) => unknown): VNode =>
  h('button', { class: 'identity-key', onClick: pressed, type: 'button' }, [
    h('i', { style: { background: swatch } }),
    h('span', { class: 'identity-caps' }, label),
    hex
  ])

/**
 * Lays out the mono caption beneath a sketch, saying its values are made up.
 */
export const madeUp = (what: string): VNode =>
  h('figcaption', { class: 'identity-note' }, `A sketch, ${what} made up`)
