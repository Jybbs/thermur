import { withBase }                       from 'vitepress'
import { type VNode, defineComponent, h } from 'vue'

import { useStage } from '../../lib/canvas/mount.ts'
import { drawLost } from '../../lib/not-found/drawing.ts'
import * as words   from '../../lib/not-found/words.ts'

/**
 * Lays out one stop on the way through the site: its ring, numbered for a
 * section and dashed for the front page, and its name as one link, with the
 * line beside it.
 */
function stop({ href, hue, label, note, numeral }: words.Stop): VNode {
  const ring = numeral === undefined
    ? h('span', { 'aria-hidden': 'true', 'class': 'not-found-ring' })
    : h('span', { class: 'not-found-ring' }, numeral)
  return h('li', hue === undefined ? { class: 'not-found-start' } : { style: { '--hue': hue } }, [
    h('a', { href: withBase(href) }, [ring, h('b', label)]),
    h('span', note)
  ])
}

/**
 * The page met at a missing address, filling the default layout's slot for it:
 * the murmuration parting around the 404 across the page, which the drawing
 * fills once the page mounts and empties once it unmounts, and beneath it the
 * title, the line on what happened, and the way through the site in reading
 * order.
 */
export default defineComponent({
  name: 'NotFoundPage',
  setup() {
    const host = useStage(drawLost)
    return () => h('div', { class: 'not-found' }, [
      h('div', { 'aria-hidden': 'true', 'class': 'not-found-stage', 'ref': host }),
      h('div', { class: 'not-found-words' }, [
        h('h1', { class: 'not-found-title' }, words.WORDS.title),
        h('p', { class: 'not-found-lead' }, words.WORDS.lead),
        h('ol', { class: 'not-found-track' }, words.trackOf().map(entry => stop(entry)))
      ])
    ])
  }
})
