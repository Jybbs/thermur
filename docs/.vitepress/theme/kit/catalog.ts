import { Radios, markup }                                     from '@thermur/kit'
import { type PropType, type VNode, defineComponent, h, ref } from 'vue'

import { PARTS, PIECES, type Piece, type Readers, pieceOf } from '../../lib/kit/pieces.ts'
import { Walk, pieceAt, useGuide }                          from '../../lib/kit/walk.ts'
import { PieceCard }                                        from './card.ts'
import { turn }                                             from './parts.ts'
import { SCENES }                                           from './scenes.ts'

/**
 * The column each piece's dot stands in along the rail, from 1, each part
 * after the first opening one column on from the last part's end, so a gap
 * parts them.
 */
const COLUMNS = PIECES.map((piece, place) => place + 1 + PARTS.indexOf(piece.part))

/**
 * The columns the rail spans, every dot's and every gap's.
 */
const SPANNED = COLUMNS.at(-1)!

/**
 * Each part's name as the rail writes it beneath its dots, its folder with a
 * capital, beside the columns its dots span.
 */
const CAPTIONS = PARTS.map((part) => {
  const columns = COLUMNS.filter((_, place) => PIECES[place]!.part === part)
  return { from: columns[0]!, name: part[0]!.toUpperCase() + part.slice(1), part, span: columns.length }
})

/**
 * The corners of the guide bracketing the piece the walk stands on.
 */
const CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const

/**
 * What a reader can do with the catalog, said once above it.
 */
const HOW = 'Back and Next walk the kit one piece at a time, and the stage sets out each part of it at work. '
          + 'Point at a dot or at a piece on the stage to read it, and press either to walk there.'

/**
 * The head above the rail, reading where the walk stands among the pieces and
 * in which part, beside Back and Next.
 */
const head = (walk: Walk, held: Piece): VNode => h('div', { class: 'kit-catalog-head' }, [
  h('span', { class: 'kit-catalog-caps' }, `${walk.place + 1} of ${PIECES.length} · ${CAPTIONS[PARTS.indexOf(held.part)]!.name}`),
  h('span', { class: 'kit-catalog-row' }, [
    turn([h('span', { 'aria-hidden': 'true' }, '← '), 'Back'], () => walk.step(-1)),
    turn(['Next', h('span', { 'aria-hidden': 'true' }, ' →')], () => walk.step(1))
  ])
])

/**
 * The rail of dots, one radio for each piece standing in its column, which a
 * press walks to and a pointer reads out, over the caption of each part.
 */
const rail = (walk: Walk, held: Piece): VNode => h('div', {
  class          : 'kit-catalog-rail',
  style          : { '--columns': SPANNED },
  onPointerleave : walk.release,
  onPointerover  : (event: Event) => {
    const key = (event.target as Element).closest('label')?.querySelector('input')?.value
    if (key) walk.hold(key)
  }
}, [
  h(Radios, {
    class    : 'kit-catalog-dots',
    label    : 'Every piece of the kit',
    options  : PIECES.map((piece, place) => ({ style: { gridColumn: String(COLUMNS[place]) }, title: piece.name, value: piece.key, words: piece.name })),
    value    : walk.picked.value,
    onChange : (key: string) => walk.picked.value = key
  }, { item: ({ value }: { value: string }) => h('i', { 'class': 'kit-catalog-dot', 'data-part': pieceOf(value).part }) }),
  h('div', { 'aria-hidden': 'true', 'class': 'kit-catalog-captions' }, CAPTIONS.map(caption => h('span', {
    'class'     : 'kit-catalog-caps',
    'data-here' : caption.part === held.part ? '' : undefined,
    'style'     : { gridColumn: `${caption.from} / span ${caption.span}` }
  }, caption.name)))
])

/**
 * The kit's catalog: the whole kit as one rail of dots grouped under its
 * parts, which Back and Next walk end to end, over a stage running the part
 * the walk stands in, a guide bracketing the piece it stands on, and beneath
 * them the card reading out that piece, or the one a reader points at
 * without moving the walk. `readers` holds the modules reading each piece,
 * which the configuration sets on the page.
 */
export default defineComponent({
  name  : 'KitCatalog',
  props : { readers: { default: () => ({}), type: Object as PropType<Readers> } },
  setup(props) {
    const walk  = new Walk(PIECES.map(piece => piece.key))
    const stage = ref<HTMLElement>()
    const guide = useGuide(stage, () => walk.picked.value)
    const pick  = (event: Event): void => {
      walk.picked.value = pieceAt(event) ?? walk.picked.value
    }
    const point = (event: Event): void => {
      const key = pieceAt(event)
      if (key) walk.hold(key)
    }
    return () => {
      const held = pieceOf(walk.picked.value)
      return h('figure', { 'class': 'kit-catalog', 'data-part': held.part }, [
        h('p', { class: 'kit-catalog-how' }, HOW),
        head(walk, held),
        rail(walk, held),
        h('div', {
          class          : 'kit-catalog-stage',
          ref            : stage,
          onFocusin      : pick,
          onPointerdown  : pick,
          onPointerleave : walk.release,
          onPointerover  : point
        }, [
          h(SCENES[held.part]!, { key: held.part }),
          h('div', {
            'aria-hidden' : 'true',
            'class'       : 'kit-catalog-guide',
            'data-placed' : guide.value ? '' : undefined,
            'style'       : guide.value && markup.placed(guide.value)
          }, CORNERS.map(corner => h('i', { class: `kit-catalog-corner kit-catalog-${corner}` })))
        ]),
        h(PieceCard, { readers: props.readers, shown: walk.shown.value })
      ])
    }
  }
})
