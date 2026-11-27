// @vitest-environment happy-dom
import { enableAutoUnmount, mount }       from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { h, nextTick }                    from 'vue'
import { renderToString }                 from 'vue/server-renderer'

import SourcesFlock          from '../../theme/sources.ts'
import { RELEASED, browser } from '../common/browser.ts'

/**
 * A page citing its first source twice and its second once, closing on the
 * Sources list the site's footnote rules write.
 */
const PAGE = '<p><a class="cite" href="#source-a" id="cite-a-1">1</a> '
           + '<a class="cite" href="#source-b" id="cite-b-1">2</a> '
           + '<a class="cite" href="#source-a" id="cite-a-2">1</a></p>'
           + '<section class="sources"><h2 class="sources-title">Sources</h2><ol class="sources-list">'
           + '<li class="source" id="source-a">'
           + '<span class="source-number"><span class="source-numeral">1</span></span>'
           + '<div class="source-body"><span class="source-backs"><a class="back" href="#cite-a-1">a</a>'
           + '<a class="back" href="#cite-a-2">b</a></span></div></li>'
           + '<li class="source" id="source-b">'
           + '<span class="source-number"><span class="source-numeral">2</span></span>'
           + '<div class="source-body"><span class="source-backs">'
           + '<a class="back" href="#cite-b-1">back</a></span></div></li>'
           + '</ol><div class="flock"></div></section>'

/**
 * Opens the page at the address fragment `hash` and mounts the flock in its
 * Sources section, as the page's Markdown sets it.
 */
function opened(hash = ''): { flock: ReturnType<typeof mount>, stand: ReturnType<typeof browser> } {
  const stand             = browser()
  location.hash           = hash
  document.body.innerHTML = PAGE
  const flock             = mount(SourcesFlock, { attachTo: '.flock' })
  return { flock, stand }
}

/**
 * Lists what the page marks current: the sources, the markers leading to a
 * current source, the marker a reader followed, and the return leading back to
 * it.
 */
const current = (): (string | null)[][] => [
  [...document.querySelectorAll('.source[data-current]')].map(each => each.id),
  [...document.querySelectorAll('a.cite[data-current]')].map(each => each.id),
  [...document.querySelectorAll('[aria-current="location"]')].map(each => each.getAttribute('href'))
]

enableAutoUnmount(afterEach)

afterEach(() => {
  document.body.innerHTML = ''
})

describe('the flock', () => {
  it('renders an empty flock on the server, so the list reads whole without it', async () => {
    expect(await renderToString(h(SourcesFlock))).toBe('<svg aria-hidden="true" class="sources-flock"></svg>')
  })

  it('draws a bird for each number once the page lays out', async () => {
    const { flock, stand } = opened()
    stand.frames.run(1)
    await nextTick()
    expect(flock.findAll('defs path')).toHaveLength(28)
    expect(flock.findAll('use').length).toBeGreaterThanOrEqual(2)
  })

  it('redraws the flock when another source becomes current', async () => {
    const { flock, stand } = opened()
    stand.frames.run(1)
    await nextTick()
    const before = flock.findAll('use').map(each => each.attributes('style'))
    document.querySelector<HTMLElement>('#cite-b-1')?.click()
    stand.frames.run(1)
    await nextTick()
    expect(flock.findAll('use').map(each => each.attributes('style'))).not.toEqual(before)
  })

  it('redraws the flock when the list changes width', async () => {
    const { stand } = opened()
    stand.frames.run(1)
    await nextTick()
    stand.reflow(document.querySelector('.sources')!, 480)
    expect(stand.held().frames).toBe(1)
  })

  it('leaves no listener, observer, or animation frame behind once it unmounts', () => {
    const { flock, stand } = opened()
    stand.reflow()
    flock.unmount()
    document.querySelector<HTMLElement>('#cite-b-1')?.click()
    expect(stand.held()).toEqual(RELEASED)
    expect(current()[0]).toEqual([])
  })
})

describe('the current source', () => {
  it('marks the source a marker leads to, every marker of it, and the return to the marker followed', () => {
    opened()
    document.querySelector<HTMLElement>('#cite-a-2')?.click()
    expect(current()).toEqual([['source-a'], ['cite-a-1', 'cite-a-2'], ['#source-a', '#cite-a-2']])
  })

  it('clears the source and the citation marked before when the reader follows another', () => {
    opened()
    document.querySelector<HTMLElement>('#cite-a-1')?.click()
    document.querySelector<HTMLElement>('#cite-b-1')?.click()
    expect(current()).toEqual([['source-b'], ['cite-b-1'], ['#source-b', '#cite-b-1']])
  })

  it('marks the source the page’s address moves to once it is open', () => {
    opened()
    location.hash = '#source-b'
    window.dispatchEvent(new HashChangeEvent('hashchange'))
    expect(current()).toEqual([['source-b'], ['cite-b-1'], []])
  })

  it('marks the marker a return leads back to', () => {
    opened()
    document.querySelector<HTMLElement>('a.back[href="#cite-b-1"]')?.click()
    expect(current()).toEqual([['source-b'], ['cite-b-1'], ['#source-b', '#cite-b-1']])
  })

  it('marks the source the page’s address names as it opens', () => {
    opened('#source-b')
    expect(current()).toEqual([['source-b'], ['cite-b-1'], []])
  })

  it('marks the marker the page’s address names as it opens, with its source', () => {
    opened('#cite-a-2')
    expect(current()).toEqual([['source-a'], ['cite-a-1', 'cite-a-2'], ['#source-a', '#cite-a-2']])
  })

  it('marks nothing when the page’s address holds a malformed escape', () => {
    opened('#%E0%A4%A')
    expect(current()).toEqual([[], [], []])
  })

  it('keeps the marker followed when the address then moves to its source', () => {
    opened()
    document.querySelector<HTMLElement>('#cite-a-1')?.click()
    location.hash = '#source-a'
    window.dispatchEvent(new HashChangeEvent('hashchange'))
    expect(current()).toEqual([['source-a'], ['cite-a-1', 'cite-a-2'], ['#source-a', '#cite-a-1']])
  })
})
