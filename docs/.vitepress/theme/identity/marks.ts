import { Slider }                              from '@thermur/kit'
import { withBase }                            from 'vitepress'
import { type VNode, defineComponent, h, ref } from 'vue'

import * as identity from '../../lib/identity.ts'
import source        from '../../lib/rendered/identity.json' with { type: 'json' }

/**
 * The kind of mark the identity files a pair of files under.
 */
type MarkKind = keyof identity.Identity['marks']

/**
 * The two color schemes, light first, as a card lays its sides out.
 */
const SCHEMES: readonly identity.Scheme[] = ['light', 'dark']
/**
 * The share of each card's width the light side opens on, in percent.
 */
const OPENING_DUSK = 50

/**
 * Lays out the card of mark `kind`: its light side and its dark side, the seam
 * between them, and its name.
 */
const markCard = (kind: MarkKind): VNode =>
  h('div', { class: `identity-card identity-card-${kind}` }, [
    ...SCHEMES.map(scheme => side(kind, scheme)),
    h('i', { 'aria-hidden': 'true', 'class': 'identity-seam' }),
    h('span', { class: 'identity-caps identity-card-name' }, kind)
  ])

/**
 * Lays out the side of a mark's card drawn for `scheme`: the file on the ground
 * of that scheme, which a press saves.
 */
function side(kind: MarkKind, scheme: identity.Scheme): VNode {
  const file = source.marks[kind][scheme]
  return h('a', {
    'aria-label' : `Save ${file}`,
    'class'      : `identity-side identity-side-${scheme}`,
    'download'   : file,
    'href'       : withBase(`/${file}`),
    'style'      : {
      backgroundColor : identity.shadeOf('ground', scheme),
      color           : identity.shadeOf('ink', scheme)
    }
  }, [
    h('img', { alt: '', src: withBase(`/${file}`) }),
    h('span', { class: 'identity-file' }, file)
  ])
}

/**
 * The marks at work: each mark drawn large in both its files at once, the light
 * file on the light ground and the dark file on the dark, divided by a seam of
 * dusk at the same place across every card, which a slider moves and which
 * settles to the middle as the page opens. A press on either side of the seam
 * saves that side's file.
 */
export default defineComponent({
  name: 'IdentityMarks',
  setup() {
    const dusk = ref(OPENING_DUSK)
    return () => h('figure', { class: 'identity-marks', style: { '--th-dusk': `${dusk.value}%` } }, [
      h('div', { class: 'identity-cards' },
        (Object.keys(source.marks) as MarkKind[]).map(kind => markCard(kind))),
      h('div', { class: 'identity-dusk' }, [
        h('span', { 'aria-hidden': 'true', 'class': 'identity-caps' }, 'Light'),
        h(Slider, {
          label    : 'Where dusk falls across the marks',
          max      : 100,
          min      : 0,
          value    : dusk.value,
          onChange : (share: number) => dusk.value = share
        }),
        h('span', { 'aria-hidden': 'true', 'class': 'identity-caps' }, 'Dark')
      ])
    ])
  }
})
