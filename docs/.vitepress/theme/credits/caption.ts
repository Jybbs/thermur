import { withBase }      from 'vitepress'
import { type VNode, h } from 'vue'

import credit from '../../lib/rendered/credit.json' with { type: 'json' }

/**
 * Lays out the caption of a figure drawn from the sample, `words` ending on a
 * link to the plume dataset's credit on the Data page.
 */
export const creditCaption = (words: string): VNode =>
  h('figcaption', { class: 'credits-caption' }, [
    `${words} `,
    h('a', { class: 'credits-link', href: withBase('/data/#credits'), title: credit.title }, 'Credits ↓')
  ])
