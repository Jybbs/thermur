import { type Ref, type VNode, h, nextTick, onBeforeUnmount, ref } from 'vue'

/**
 * What a widget copies through: the copy itself, and the words a screen reader
 * hears once a copy lands.
 */
export interface Copier {
  copy : (text: string, event: Event) => Promise<void>
  said : Ref<string>
}

/**
 * A color written as a six-digit hex.
 */
const HEX = /^#[\da-f]{6}$/iu
/**
 * The milliseconds a pressed control reads as copied.
 */
export const COPY_HOLD = 1400

/**
 * Lays out `said`, the words a screen reader hears once a copy lands, out of
 * sight.
 */
export const heard = (said: Ref<string>): VNode =>
  h('span', { 'aria-live': 'polite', 'class': 'visually-hidden' }, said.value)

/**
 * Copies a value from a widget's control and marks the control as copied for
 * `COPY_HOLD` milliseconds through `data-copied`, restarting the mark on a
 * second press, with a copied hex set as `--copied` for the confirmation to
 * draw in. Each copy writes what it copied into `said`, or that it could not
 * copy where the clipboard refuses the text, emptying `said` first so a screen
 * reader hears a second copy of the same value, and every hold still running
 * ends as the widget unmounts.
 */
export function useCopy(): Copier {
  const said   = ref('')
  const timers = new Map<HTMLElement, number>()
  onBeforeUnmount(() => {
    for (const timer of timers.values()) clearTimeout(timer)
  })
  const say    = async (words: string): Promise<void> => {
    said.value = ''
    await nextTick()
    said.value = words
  }
  const copy   = async (text: string, event: Event): Promise<void> => {
    const node = event.currentTarget as HTMLElement
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      await say(`Could not copy ${text}`)
      return
    }
    clearTimeout(timers.get(node))
    delete node.dataset.copied
    if (HEX.test(text)) node.style.setProperty('--copied', text)
    else node.style.removeProperty('--copied')
    // Reading the width lays the page out again, so the mark set next starts
    // its confirmation over.
    void node.offsetWidth
    node.dataset.copied = ''
    timers.set(node, window.setTimeout(() => {
      delete node.dataset.copied
      timers.delete(node)
    }, COPY_HOLD))
    await say(`Copied ${text}`)
  }
  return { copy, said }
}
