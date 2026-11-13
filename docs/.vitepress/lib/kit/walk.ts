import { markup } from '@thermur/kit'
import { type Ref, type ShallowRef, computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'

/**
 * Where the catalog stands among the kit's pieces: the piece the walk has
 * reached, the one a reader points at without moving the walk, and so the one
 * the card reads out, the pointed one while a reader points and the reached
 * one otherwise.
 */
export class Walk {
  readonly keys     : readonly string[]
  readonly picked   : Ref<string>
  readonly pointing = ref<string>()
  readonly shown    = computed(() => this.pointing.value ?? this.picked.value)

  constructor(keys: readonly string[]) {
    this.keys   = keys
    this.picked = ref(keys[0]!)
  }

  /**
   * Counts the reached piece's place among every piece, from 0.
   */
  get place(): number {
    return this.keys.indexOf(this.picked.value)
  }

  /**
   * Walks `by` pieces on from the reached one, wrapping at either end, and
   * lets go of the piece a reader points at, so the card reads the reached
   * one.
   */
  step(by: number): void {
    this.picked.value   = this.keys[(this.place + by + this.keys.length) % this.keys.length]!
    this.pointing.value = undefined
  }

  /**
   * Points at `key` until `release` lets go of it.
   */
  readonly hold = (key: string): void => {
    this.pointing.value = key
  }

  /**
   * Lets go of the piece a reader points at.
   */
  readonly release = (): void => {
    this.pointing.value = undefined
  }
}

/**
 * Reads the key a pointer, a press, or the focus lands on through `event`,
 * meaning the `data-piece` of the nearest element above its target that names
 * one, or nothing where none does.
 */
export const pieceAt = (event: Event): string | undefined =>
  (event.target as Element | null)?.closest<HTMLElement>('[data-piece]')?.dataset.piece

/**
 * Measures where the piece `key` names stands within the element `stage`
 * holds, as a box from the stage's top left corner, again whenever the key
 * changes or the stage changes size, or nothing until the piece is found. The
 * observer is released once the component unmounts.
 */
export function useGuide(stage: Ref<HTMLElement | undefined>, key: () => string): ShallowRef<markup.Spot | undefined> {
  const spot     = shallowRef<markup.Spot>()
  const measure  = (): void => {
    const piece = stage.value?.querySelector(`[data-piece="${key()}"]`)
    if (!stage.value || !piece) {
      spot.value = undefined
      return
    }
    const framed = stage.value.getBoundingClientRect()
    const box    = piece.getBoundingClientRect()
    spot.value   = { height: box.height, left: box.left - framed.left, top: box.top - framed.top, width: box.width }
  }
  const observer = new ResizeObserver(measure)
  onMounted(() => {
    observer.observe(stage.value!)
    watch(key, measure, { flush: 'post', immediate: true })
  })
  onBeforeUnmount(() => observer.disconnect())
  return spot
}
