import { Toggle, heard, useCopy } from '@thermur/kit'
import { type VNode, h }          from 'vue'

import { heatRamp } from '../../lib/identity.ts'
import source       from '../../lib/rendered/identity.json' with { type: 'json' }

/**
 * The heat ramp's steps as hexes, lightest first, which the copies take.
 */
export const STEPS = heatRamp(source)

/**
 * Sets `children` in the element one piece of a scene stands in, named for
 * the piece's key so the walk can find it, point at it, and bracket it.
 */
export const part = (key: string, children: VNode | string | readonly (VNode | string)[], wide = false): VNode =>
  h('div', { 'class': ['kit-catalog-part', { 'kit-catalog-wide': wide }], 'data-piece': key }, [children].flat())

/**
 * A button reading `words`, a press on which runs `pressed`.
 */
export const turn = (words: VNode | string | (VNode | string)[], pressed: (event: Event) => unknown): VNode =>
  h('button', { class: 'kit-catalog-turn', onClick: pressed, type: 'button' }, words)

/**
 * A toggle playing or holding a run, its face the same triangle and bars the
 * page's pause switch shows.
 */
export const playing = (label: string, pressed: boolean, changed: (on: boolean) => void): VNode =>
  h(Toggle, { class: 'kit-catalog-play', label, onChange: changed, pressed }, {
    default: ({ pressed: on }: { pressed: boolean }) =>
      h('span', { 'aria-hidden': 'true', 'class': `motion-pause-${on ? 'bars' : 'play'}` })
  })

/**
 * A row of the heat ramp's steps, each a button copying its hex and burning
 * the kit's fuse beneath it in that hex, beside the words a screen reader
 * hears once a copy lands.
 */
export function useSteps(): () => VNode {
  const { copy, said } = useCopy()
  return () => h('div', { class: 'kit-catalog-steps' }, [
    ...STEPS.map((hex, place) => h('button', {
      'aria-label' : `Copy step ${place + 1}, ${hex}`,
      'class'      : 'kit-catalog-step',
      'type'       : 'button',
      'onClick'    : (event: Event) => copy(hex, event)
    }, [
      h('i', { style: { background: `var(--th-heat-${place + 1})` } }),
      h('span', { class: 'kit-fuse' }, hex)
    ])),
    heard(said)
  ])
}
