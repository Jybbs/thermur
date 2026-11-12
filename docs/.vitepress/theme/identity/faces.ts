import { type Ref, type VNode, defineComponent, h, onBeforeUnmount, reactive, ref } from 'vue'

import { type Identity, fontStack } from '../../lib/identity.ts'
import source                       from '../../lib/rendered/identity.json' with { type: 'json' }
import { tween }                    from '../../lib/rules/motion.ts'
import { type Copier, useCopy }     from '../../lib/specimen/copy.ts'
import * as sketch                  from '../../lib/specimen/sketch.ts'
import { heard, madeUp }            from './parts.ts'

/**
 * A face the identity sets, beside the kind of text it is set for.
 */
type Face = [TextKind, Identity['faces'][TextKind]]

/**
 * The kind of text a face is set for.
 */
type TextKind = keyof Identity['faces']

/**
 * The gauges of the flight's panel, each a reading beside its name and unit.
 */
const GAUGES = [
  { name: 'Altitude', reading: 'altitude', unit: 'm' },
  { name: 'Air', reading: 'air', unit: 'K' },
  { name: 'Split', reading: 'split', unit: '%' }
] as const
/**
 * The second of the flight the panel opens on.
 */
const OPENING_SECOND = 41

/**
 * Reads the kind of text the part of the panel under the pointer is set in, or
 * `null` over a part set in none.
 */
const faceOf = (event: Event): TextKind | null =>
  ((event.target as Element).closest<HTMLElement>('[data-face]')?.dataset.face as TextKind | undefined)
  ?? null

/**
 * Lays out the row of the face `face`, naming it in its own letters beside the
 * text it sets, each weight it ships set at that weight, its italic where it
 * has one, and its stack, which a press copies. Pointing at the row or moving
 * the focus onto it holds the face in `held`.
 */
function row(copy: Copier['copy'], [kind, face]: Face, held: Ref<TextKind | null>): VNode {
  const stack = fontStack(face)
  return h('button', {
    'class'          : 'identity-face',
    'data-held'      : held.value === kind ? '' : undefined,
    'onClick'        : (event: Event) => copy(stack, event),
    'onFocus'        : () => (held.value = kind),
    'onPointerenter' : () => (held.value = kind),
    'style'          : { fontFamily: stack },
    'type'           : 'button'
  }, [
    h('span', { class: 'identity-face-name' }, face.family),
    h('span', { class: 'identity-caps' }, kind),
    h('span', { class: 'identity-weights' }, [
      ...face.weights.map(weight => h('span', { style: { fontWeight: weight } }, weight)),
      face.italic ? h('i', 'Italic') : null
    ]),
    h('span', { class: 'identity-hex' }, stack)
  ])
}

/**
 * Lays out the flight's panel at `second`, its gauges at `shown`, each part
 * dimmed while `held` holds a face other than the one it is set in, and its
 * scrubber running `moved` on each move.
 */
function panel(
  held   : Ref<TextKind | null>,
  moved  : (event: Event) => void,
  second : number,
  shown  : sketch.PanelReadings
): VNode {
  const part                   = (name: string, kind: TextKind): Record<string, unknown> => ({
    'class'     : [name, { 'identity-dim': held.value !== null && held.value !== kind }],
    'data-face' : kind
  })
  const [, plain, quoted, end] = sketch.entryAt(second)
  return h('div', {
    class          : 'identity-panel',
    onPointerleave : () => (held.value = null),
    onPointerover  : (event: PointerEvent) => (held.value = faceOf(event))
  }, [
    h('h3', part('identity-title', 'display'), 'A Flight Through the Plume'),
    h('div', part('identity-gauges', 'mono'), GAUGES.map(({ name, reading, unit }) =>
      h('div', { class: 'identity-gauge' }, [
        h('span', { class: 'identity-caps' }, name),
        h('span', { class: 'identity-reading' }, Math.round(shown[reading])),
        h('span', { class: 'identity-unit' }, unit)
      ]))),
    h('p', part('identity-log', 'body'), [plain, h('em', quoted), end]),
    h('label', part('identity-scrub', 'mono'), [
      h('span', { class: 'identity-caps' }, 'Time'),
      h('input', {
        'aria-label' : 'Time into the flight',
        'max'        : sketch.SPAN,
        'min'        : 0,
        'onInput'    : moved,
        'step'       : 1,
        'type'       : 'range',
        'value'      : second
      }),
      h('span', { class: 'identity-reading' }, `${second} s`)
    ])
  ])
}

/**
 * The faces at work: a sketch of a flight's panel, its title in the display
 * face, a log in the body face that a time scrubber moves through, and each
 * reading in the mono face, each figure settling to its new value, over a row
 * per face naming it in its own letters beside each weight and style it ships.
 * Pointing at a row or at the panel picks out that face's parts of the panel,
 * and a press on a row copies its stack.
 */
export default defineComponent({
  name: 'IdentityFaces',
  setup() {
    const { copy, said } = useCopy()
    const held           = ref<TextKind | null>(null)
    const second         = ref(OPENING_SECOND)
    const shown          = reactive(sketch.readingsAt(OPENING_SECOND))
    const faces          = Object.entries(source.faces) as Face[]
    let settling: (() => void) | undefined
    onBeforeUnmount(() => settling?.())
    const moved          = (event: Event): void => {
      const from   = { ...shown }
      second.value = (event.target as HTMLInputElement).valueAsNumber
      const to     = sketch.readingsAt(second.value)
      settling?.()
      settling     = tween(share => {
        for (const { reading } of GAUGES)
          shown[reading] = from[reading] + (to[reading] - from[reading]) * share
      }, 2)
    }
    return () => h('figure', { class: 'identity-faces' }, [
      panel(held, moved, second.value, shown),
      madeUp('its readings'),
      h('div', { class: 'identity-face-list', onFocusout: () => (held.value = null) },
        faces.map(face => row(copy, face, held))),
      heard(said)
    ])
  }
})
