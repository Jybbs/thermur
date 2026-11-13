// @vitest-environment happy-dom
import { type markup }                               from '@thermur/kit'
import { type VueWrapper, enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi }       from 'vitest'
import { type Ref, type ShallowRef, defineComponent, h, nextTick, ref } from 'vue'

import { Walk, pieceAt, useGuide } from '../../../lib/kit/walk.ts'
import { RELEASED, browser }       from '../../common/browser.ts'

enableAutoUnmount(afterEach)

describe('Walk', () => {
  it('steps on and back from the first piece, wrapping at either end', () => {
    const walk = new Walk(['a', 'b', 'c'])
    expect([walk.picked.value, walk.place]).toEqual(['a', 0])
    walk.step(-1)
    expect([walk.picked.value, walk.place]).toEqual(['c', 2])
    walk.step(1)
    walk.step(1)
    expect(walk.picked.value).toBe('b')
  })

  it('shows the piece a reader points at until it is let go or the walk steps', () => {
    const walk = new Walk(['a', 'b', 'c'])
    walk.hold('c')
    expect([walk.shown.value, walk.picked.value]).toEqual(['c', 'a'])
    walk.release()
    expect(walk.shown.value).toBe('a')
    walk.hold('c')
    walk.step(1)
    expect(walk.shown.value).toBe('b')
  })
})

/**
 * Reads the piece an event targeting the element `selector` finds lands on.
 */
const at = (selector: string): string | undefined => pieceAt({ target: document.querySelector(selector) } as unknown as Event)

describe('pieceAt', () => {
  it('reads the piece named nearest above the target, or nothing', () => {
    document.body.innerHTML = '<div data-piece="slider"><button></button></div><p></p>'
    expect(at('button')).toBe('slider')
    expect(at('p')).toBeUndefined()
    expect(pieceAt({ target: null } as unknown as Event)).toBeUndefined()
  })
})

/**
 * Mounts a stage holding the pieces `one` and `two`, guided to the key
 * `key` holds, each element carrying how far it stands from the page's
 * corner in `data-offset`.
 */
function guided(): { guide: ShallowRef<markup.Spot | undefined>, key: Ref<string>, stage: VueWrapper } {
  const key   = ref('one')
  let guide!: ShallowRef<markup.Spot | undefined>
  const stage = mount(defineComponent({
    setup() {
      const held = ref<HTMLElement>()
      guide      = useGuide(held, () => key.value)
      return () => h('div', { 'data-offset': 0, 'ref': held }, [
        h('i', { 'data-offset': 10, 'data-piece': 'one' }),
        h('i', { 'data-offset': 40, 'data-piece': 'two' })
      ])
    }
  }), { attachTo: document.body })
  return { guide, key, stage }
}

describe('useGuide', () => {
  it('measures the guided piece from the stage’s corner, again as the key and the stage change', async () => {
    const stand = browser()
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function boxed(this: HTMLElement) {
      const offset = Number(this.dataset.offset)
      return { height: 8, left: 100 + offset, top: 50 + offset * 2, width: 30 } as DOMRect
    })
    const { guide, key } = guided()
    expect(guide.value).toEqual({ height: 8, left: 10, top: 20, width: 30 })
    key.value = 'two'
    await nextTick()
    expect(guide.value).toEqual({ height: 8, left: 40, top: 80, width: 30 })
    key.value = 'none'
    stand.reflow()
    expect(guide.value).toBeUndefined()
  })

  it('releases its observer once the stage unmounts', () => {
    const stand     = browser()
    const { stage } = guided()
    expect(stand.held().observed).toBe(1)
    stage.unmount()
    expect(stand.held()).toEqual(RELEASED)
  })
})
