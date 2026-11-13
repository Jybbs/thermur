import { type Copier, heard, idle, tween, useCopy }                                                        from '@thermur/kit'
import { type PropType, type VNode, defineComponent, h, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'

import { lerp }                                                       from '../../lib/common.ts'
import { PIECES, type Piece, type Readers, SOURCES, lineOf, pieceOf } from '../../lib/kit/pieces.ts'
import { EXAMPLES }                                                   from './examples.ts'
import { turn }                                                       from './parts.ts'

/**
 * Sets one slot holding what `draw` writes for every piece, every one in the
 * same cell so the slot stands as wide and as tall as the largest of them,
 * and only the one for `shown` seen and reachable, so a walk moves nothing
 * around the slot.
 */
const stacked = (shown: string, draw: (piece: Piece) => VNode | VNode[]): VNode =>
  h('div', { class: 'kit-card-stack' }, PIECES.map(piece => h('div', {
    'aria-hidden' : piece.key === shown ? undefined : 'true',
    'data-shown'  : piece.key === shown ? '' : undefined,
    'inert'       : piece.key !== shown,
    'key'         : piece.key
  }, draw(piece))))

/**
 * One fact the opened card sets out, its label in the margin.
 */
const fact = (label: string, body: VNode | string | (VNode | string)[]): VNode =>
  h('div', { class: 'kit-card-fact' }, [h('span', { class: 'kit-card-label' }, label), h('div', { class: 'kit-card-body' }, body)])

/**
 * Each example of `piece` at work, beside what a reader sees it do.
 */
const examples = (piece: Piece): VNode[] => piece.examples.map(example => h('div', { class: 'kit-card-example', key: example.name }, [
  h('span', { class: 'kit-card-caps' }, example.name),
  h(EXAMPLES[piece.key]![example.name]!),
  h('p', { class: 'kit-card-says' }, example.says)
]))

/**
 * Loads the source of each file of `piece` into `loaded`, once apiece.
 */
const load = (piece: Piece, loaded: Map<string, string>): void => {
  for (const path of piece.sources) {
    if (!loaded.has(path)) void SOURCES[path]!().then(text => loaded.set(path, text))
  }
}

/**
 * What the opened card carries for `piece`: what it replaces, its examples at
 * work, the modules `readers` lists reading it, and the line taking it up,
 * which `copy` copies, over its source as `loaded` holds it once opened.
 */
function carried(piece: Piece, readers: Readers, copy: Copier['copy'], loaded: Map<string, string>): VNode[] {
  const read = readers[piece.key] ?? []
  return [
    fact('In place of', h('p', { class: 'kit-card-says' }, piece.replaces)),
    fact('At work', examples(piece)),
    fact('Read by', read.length === 0
      ? h('p', { class: 'kit-card-says' }, 'No module reads it yet.')
      : h('ul', { class: 'kit-card-paths' }, read.map(path => h('li', h('code', path))))),
    fact('To take it up', [
      h('div', { class: 'kit-catalog-row' }, [
        h('pre', { class: 'kit-card-line' }, h('code', lineOf(piece))),
        turn(h('span', { class: 'kit-fuse' }, 'Copy'), (event: Event) => copy(lineOf(piece), event))
      ]),
      h('details', { class: 'kit-card-source', onToggle: () => load(piece, loaded) }, [
        h('summary', { class: 'kit-card-caps' }, 'Its source'),
        ...piece.sources.map(path => h('div', { class: 'kit-card-file', key: path }, [
          h('code', `docs/kit/src/${path}`),
          h('pre', h('code', loaded.get(path) ?? 'Loading the source'))
        ]))
      ])
    ])
  ]
}

/**
 * The card beneath the stage, reading out the piece `shown`: the Zag.js
 * machine it runs on over how many of the site's modules read it, in the hue
 * of its part, beside its name and when a widget uses it, and a fold
 * opening on what it replaces, its examples at work, the modules reading it,
 * and the line a widget writes to take it up, which a press copies, over its
 * source, loaded once a reader opens it.
 */
export const PieceCard = defineComponent({
  name  : 'KitPieceCard',
  props : {
    readers : { required: true, type: Object as PropType<Readers> },
    shown   : { required: true, type: String }
  },
  setup(props) {
    const { copy, said } = useCopy()
    const loaded         = reactive(new Map<string, string>())
    const counted        = ref(0)
    let stop             = idle
    const count          = (): number => props.readers[props.shown]?.length ?? 0
    const ease           = (to: number): void => {
      stop()
      const from = counted.value
      stop       = tween(share => counted.value = lerp(from, to, share))
    }
    watch(count, ease)
    onMounted(() => ease(count()))
    onBeforeUnmount(() => stop())
    return () => {
      const piece = pieceOf(props.shown)
      return h('section', { 'aria-label': piece.name, 'class': 'kit-card', 'data-part': piece.part }, [
        h('div', { class: 'kit-card-face' }, [
          h('div', { class: 'kit-card-swatch' }, [
            stacked(props.shown, one => h('span', { class: 'kit-card-caps' }, one.machine === null ? 'Its own' : `Zag ${one.machine}`)),
            h('span', { class: 'kit-card-count' }, [
              h('span', { class: 'kit-card-figure' }, String(Math.round(counted.value))),
              h('span', { class: 'kit-card-caps' }, count() === 1 ? 'module reads it' : 'modules read it')
            ])
          ]),
          h('div', { class: 'kit-card-main' }, [
            stacked(props.shown, one => h('span', { class: 'kit-card-name' }, one.name)),
            stacked(props.shown, one => h('p', { class: 'kit-card-when' }, one.when))
          ])
        ]),
        h('details', { class: 'kit-card-fold' }, [
          h('summary', { class: 'kit-card-more' }, 'Everything it carries'),
          stacked(props.shown, one => carried(one, props.readers, copy, loaded))
        ]),
        heard(said)
      ])
    }
  }
})
