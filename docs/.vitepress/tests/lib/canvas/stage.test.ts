// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest'

import type { Geometry, Part }                         from '../../../lib/canvas/part.ts'
import { Stage }                                       from '../../../lib/canvas/stage.ts'
import { type Browser, RELEASED, box, browser, point } from '../../common/browser.ts'
import { LAYOUT, front }                               from '../../common/front.ts'

/**
 * A stage over the stand-in browser holding one spied part, beside the part and
 * the stand-ins.
 */
interface Staged {
  drawn : Part & { painted: Geometry[] }
  stage : Stage
  stand : Browser
}

/**
 * A stage set in the page's flow.
 */
const FLOWING = { pinned: false } as const
/**
 * A stage pinned behind the whole page.
 */
const PINNED = { pinned: true } as const

/**
 * Copies the geometry a paint read, which the stage updates in place.
 */
const copied = (g: Geometry): Geometry =>
  ({ ...g, hero: { ...g.hero }, pointer: g.pointer && [g.pointer[0], g.pointer[1]] })

/**
 * Builds a part whose every call is a spy, holding a copy of the geometry each
 * paint read.
 */
function part(): Part & { painted: Geometry[] } {
  const painted: Geometry[] = []
  return {
    fit     : vi.fn<(g: Geometry) => void>(),
    paint   : vi.fn<Part['paint']>((_, g) => painted.push(copied(g))),
    painted : painted,
    step    : vi.fn<(dt: number, g: Geometry) => void>()
  }
}

/**
 * Builds a stage holding one spied part over the stand-in browser, pinned
 * behind a front page or set in the flow of the page's body as `mount` sets,
 * for a reader who prefers reduced motion where `reduced` holds.
 */
function staged(mount: { pinned: boolean } = PINNED, reduced = false): Staged {
  const stand = browser(reduced)
  const drawn = part()
  const stage = new Stage(mount.pinned ? front() : document.body, [drawn], mount)
  return { drawn, stage, stand }
}

describe('the stage’s fit', () => {
  it('fits and paints its parts on the page as it stands once mounted', () => {
    const { drawn, stage } = staged()
    expect(drawn.fit).toHaveBeenCalledOnce()
    expect(drawn.painted[0]).toMatchObject({
      doc  : { left: LAYOUT.column.left, right: LAYOUT.column.right },
      hero : { bottom: LAYOUT.hero.height, h: LAYOUT.hero.height, top: 0 },
      page : LAYOUT.page,
      w    : 1200
    })
    stage.destroy()
  })

  it('measures a page with no hero and no reading column as a hero of no height and no column', () => {
    browser()
    window.scrollTo(0, 0)
    const drawn = part()
    const stage = new Stage(box('front-sky'), [drawn], PINNED)
    expect(drawn.painted[0]).toMatchObject({ doc: null, hero: { bottom: 0, h: 0, top: 0 } })
    stage.destroy()
  })

  it('fits again once the page reflows while it stands still, reading the page’s new length', () => {
    const { drawn, stage, stand } = staged(PINNED, true)
    vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockReturnValue(LAYOUT.page + 300)
    stand.reflow()
    expect(drawn.fit).toHaveBeenCalledTimes(2)
    expect(drawn.painted.at(-1)!.page).toBe(LAYOUT.page + 300)
    stage.destroy()
  })

  it('holds its canvas at twice its box on a screen of a higher pixel ratio', () => {
    vi.stubGlobal('devicePixelRatio', 3)
    const { stage } = staged()
    const canvas    = document.querySelector('canvas')!
    expect([canvas.width, canvas.height]).toEqual([2400, 1600])
    stage.destroy()
  })

  it('fits nothing to a canvas of no size and paints nothing on a page not yet fitted', () => {
    browser(true).resize(0, 0)
    const drawn = part()
    const stage = new Stage(document.body, [drawn], PINNED)
    window.dispatchEvent(new Event('scroll'))
    expect(drawn.fit).not.toHaveBeenCalled()
    expect(drawn.painted).toEqual([])
    stage.destroy()
  })
})

describe('the stage’s motion', () => {
  it('steps and paints its parts on each frame while it is seen and motion runs', () => {
    const { drawn, stage, stand } = staged()
    stand.see(true)
    stand.frames.run(3, 1000)
    expect(drawn.step).toHaveBeenCalledTimes(3)
    expect(drawn.painted).toHaveLength(4)
    expect(vi.mocked(drawn.step!).mock.calls.map(([dt]) => dt)).toEqual([1 / 60, 0.05, 0.05])
    const painted = drawn.painted.length
    point(window, { clientX: 40, clientY: 50 })
    document.documentElement.dispatchEvent(new Event('pointerleave'))
    window.dispatchEvent(new Event('scroll'))
    expect(drawn.painted).toHaveLength(painted)
    stand.see(false)
    stand.frames.run(2)
    expect(drawn.step).toHaveBeenCalledTimes(3)
    stage.destroy()
  })
})

describe('the stage standing still', () => {
  it('draws one frame for a reader who prefers reduced motion and schedules none', () => {
    const { drawn, stage, stand } = staged(PINNED, true)
    stand.see(true)
    expect(drawn.paint).toHaveBeenCalledOnce()
    expect(stand.held().frames).toBe(0)
    expect(drawn.painted[0]!.still).toBe(true)
    stage.destroy()
  })

  it('repaints on a scroll, a pointer, and a scheme while still, its pointer following the scroll', () => {
    const { drawn, stage, stand } = staged(PINNED, true)
    window.scrollTo(0, 120)
    window.dispatchEvent(new Event('scroll'))
    point(window, { clientX: 40, clientY: 50 })
    document.documentElement.classList.add('dark')
    stand.mutate()
    window.scrollTo(0, 200)
    window.dispatchEvent(new Event('scroll'))
    document.documentElement.dispatchEvent(new Event('pointerleave'))
    expect(drawn.painted.map(g => [g.dy, g.pointer, g.scheme])).toEqual([
      [0, null, 'light'], [120, null, 'light'], [120, [40, 170], 'light'], [120, [40, 170], 'dark'],
      [200, [40, 250], 'dark'], [200, null, 'dark']
    ])
    document.documentElement.classList.remove('dark')
    window.scrollTo(0, 0)
    stage.destroy()
  })
})

describe('the stage standing still and moving again', () => {
  it('stands still while the page’s pause is on and runs again once it is off', () => {
    const { drawn, stage, stand } = staged()
    stand.see(true)
    document.documentElement.dataset.paused = 'true'
    stand.frames.run(2)
    expect(drawn.step).toHaveBeenCalledTimes(0)
    delete document.documentElement.dataset.paused
    stand.mutate()
    stand.frames.run(2)
    expect(drawn.step).toHaveBeenCalledTimes(2)
    stage.destroy()
  })

  it('stands still while the reader prefers reduced motion and runs again once the preference lifts', () => {
    const { drawn, stage, stand } = staged(PINNED, true)
    stand.see(true)
    stand.frames.run(2)
    expect(drawn.step).toHaveBeenCalledTimes(0)
    stand.motion.matches = false
    for (const listener of stand.motion.listeners) listener()
    stand.frames.run(2)
    expect(drawn.step).toHaveBeenCalledTimes(2)
    stage.destroy()
  })
})

describe('a stage set in the page’s flow', () => {
  it('reads the pointer over its canvas in the canvas’s own coordinates, until it leaves', () => {
    const { drawn, stage } = staged(FLOWING, true)
    const canvas           = document.querySelector('canvas')!
    window.scrollTo(0, 120)
    point(window, { clientX: 400, clientY: 400 })
    point(canvas, { offsetX: 40, offsetY: 50 })
    canvas.dispatchEvent(new Event('pointerleave'))
    expect(drawn.painted.map(g => g.pointer)).toEqual([null, [40, 50], null])
    window.scrollTo(0, 0)
    stage.destroy()
  })

  it('hands a click on its canvas to each part that takes one, repainting while it stands still', () => {
    browser(true)
    const drawn = { ...part(), press: vi.fn<(g: Geometry) => void>() }
    const stage = new Stage(document.body, [drawn, part()], FLOWING)
    document.querySelector('canvas')!.dispatchEvent(new Event('click'))
    expect(drawn.press).toHaveBeenCalledOnce()
    expect(drawn.painted).toHaveLength(2)
    stage.destroy()
  })

  it('fits its parts again on a refit, and none once its canvas is gone', () => {
    const { drawn, stage } = staged(FLOWING)
    stage.refit()
    expect(drawn.fit).toHaveBeenCalledTimes(2)
    stage.destroy()
    stage.refit()
    expect(drawn.fit).toHaveBeenCalledTimes(2)
  })
})

describe('the stage’s teardown', () => {
  it.for([['pinned behind the page', PINNED], ['set in the page’s flow', FLOWING]] as const)(
    'leaves no listener, observer, frame, or canvas behind once a stage %s is destroyed',
    ([, mount]) => {
      const { stage, stand } = staged(mount)
      stand.see(true)
      stand.frames.run(2)
      expect(stand.held().listeners).toBeGreaterThan(0)
      stage.destroy()
      expect(stand.held()).toEqual(RELEASED)
    }
  )

  it('hands no event to its parts once destroyed', () => {
    browser(true)
    const drawn  = { ...part(), press: vi.fn<(g: Geometry) => void>() }
    const stage  = new Stage(document.body, [drawn], FLOWING)
    const canvas = document.querySelector('canvas')!
    stage.destroy()
    point(canvas, { offsetX: 40, offsetY: 50 })
    canvas.dispatchEvent(new Event('click'))
    window.dispatchEvent(new Event('scroll'))
    expect(drawn.press).not.toHaveBeenCalled()
    expect(drawn.painted).toHaveLength(1)
  })
})
