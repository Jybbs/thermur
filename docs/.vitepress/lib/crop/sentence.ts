import { type Ref, onBeforeUnmount, onMounted, ref } from 'vue'

import source                from '../rendered/identity.json' with { type: 'json' }
import { type Kept, MOST_M } from './kept.ts'

/**
 * One option of a menu: what it sets on the strip, the line it writes under
 * `[tool.thermur.crop]`, none for a default, and the words it reads as.
 */
interface Choice {
  apply : () => void
  line  : string | null
  name  : string
}

/**
 * One menu of the sentence: the choices it offers and the name a screen reader
 * gives it.
 */
export interface Menu {
  choices : readonly Choice[]
  label   : string
}

/**
 * What the lines read while every menu stands at its default.
 */
export const DEFAULTS = 'These are the defaults, so there is nothing to set.'
/**
 * The lowest height the menus keep, in meters.
 */
const LOWEST_M = 200

/**
 * Sets the sentence's height to what its widest wording takes at the width it
 * has, leaving each menu at the pick it held.
 */
function hold(sentence: HTMLElement): void {
  const selects = [...sentence.querySelectorAll('select')]
  const picks   = selects.map(({ selectedIndex }) => selectedIndex)
  sentence.style.minHeight = ''
  for (const select of selects) select.selectedIndex = widest(select)
  sentence.style.minHeight = `${sentence.getBoundingClientRect().height}px`
  for (const [index, select] of selects.entries()) select.selectedIndex = picks[index]!
}

/**
 * Writes a height as the sentence reads it.
 */
const meters = (height: number): string => `${height.toLocaleString('en-US')} m`

/**
 * Finds the place of the option of `select` that lays out widest, each option
 * standing in the menu in turn.
 */
function widest(select: HTMLSelectElement): number {
  const widths = Array.from(select.options, ({ index }) => {
    select.selectedIndex = index
    return select.getBoundingClientRect().width
  })
  return widths.indexOf(Math.max(...widths))
}

/**
 * Writes the lines the picks set under the table's name, or nothing where every
 * pick is a default.
 *
 * Args:
 *   menus : The menus the sentence holds.
 *   picks : The place of the choice each menu stands at.
 */
export function linesOf(menus: readonly Menu[], picks: readonly number[]): string | null {
  const lines = menus.flatMap(({ choices }, index) => choices[picks[index]!]!.line ?? [])
  return lines.length > 0 ? ['[tool.thermur.crop]', ...lines].join('\n') : null
}

/**
 * Builds the three menus from what `kept` holds, so each line they write is the
 * record's own arithmetic and each choice moves the part of `kept` it names.
 */
export function menusOf(kept: Kept): readonly Menu[] {
  const { east, first, model, reach, top, west } = kept
  const half                                     = (model.length_m / 2).toFixed(1)
  const last                                     = model.times_s.length - 1
  const whole                                    = model.length_m.toFixed(1)
  const along = (from: number, to: number) => (): void => {
    west.target = from
    east.target = to
  }
  const since = (snapshot: number) => (): void => {
    first.target = snapshot
  }
  const up    = (height: number) => (): void => {
    top.target = height
  }
  const levels = (height: number): string => `levels = ${(height - model.floor_m) / model.spacing_m + 1}`
  return [
    {
      label   : 'Where along the ground',
      choices : [
        { apply: along(0, 1), line: null, name: 'the whole domain' },
        { apply: along(0, 0.5), line: `x_m = [0.0, ${half}]`, name: 'the western half' },
        { apply: along(0.5, 1), line: `x_m = [${half}, ${whole}]`, name: 'the eastern half' }
      ]
    },
    {
      label   : 'When',
      choices : [
        { apply: since(0), line: null, name: 'every snapshot' },
        {
          apply : since(last - 1),
          line  : `window_s = [${model.times_s[last - 1]!.toFixed(1)}, ${model.times_s[last]!.toFixed(1)}]`,
          name  : 'the last two snapshots'
        }
      ]
    },
    {
      label   : 'How high',
      choices : [
        { apply: up(reach), line: null, name: meters(reach) },
        { apply: up(LOWEST_M), line: levels(LOWEST_M), name: meters(LOWEST_M) },
        { apply: up(MOST_M), line: levels(MOST_M), name: meters(MOST_M) }
      ]
    }
  ]
}

/**
 * Builds the sentence's element and holds it at the height its widest wording
 * takes, measured once it mounts, again whenever its width changes, and again
 * once the face it reads in has loaded, so no pick moves the page. The observer
 * disconnects once the component unmounts.
 */
export function useHeldHeight(): Ref<HTMLElement | undefined> {
  const sentence = ref<HTMLElement>()
  let observer   : ResizeObserver | undefined
  onMounted(() => {
    const element = sentence.value!
    let width     = 0
    observer      = new ResizeObserver(() => {
      const now = element.getBoundingClientRect().width
      if (now === width) return
      width = now
      hold(element)
    })
    observer.observe(element)
    void document.fonts.load(`400 1em '${source.faces.body.family}'`).then(() => hold(element))
  })
  onBeforeUnmount(() => observer?.disconnect())
  return sentence
}
