import { seriesProperty }                    from '../identity.ts'
import { SECTIONS, type Section, numeralOf } from '../sections.ts'

/**
 * One stop on the way through the site the page met at a missing address lays
 * out: the address it links beneath the site's base, its name and the line
 * beside it, and for a section, its place in reading order as two digits and
 * the custom property its ring's hue is read from.
 */
export interface Stop {
  href     : string
  label    : string
  note     : string
  hue?     : string
  numeral? : string
}

/**
 * The words the page met at a missing address sets beneath its drawing: the
 * title, the line on what happened, and the front page's stop, where the way
 * through the site starts.
 */
export const WORDS = {
  lead  : 'No page sits at this address. It may have moved, or the link that led here may be mistyped.',
  start : { href: '/', label: 'The front page', note: 'Where the reading starts.' },
  title : 'Page Not Found'
} as const

/**
 * Lists the way through the site in reading order, the front page where it
 * starts and then each section of `sections` at its overview, numbered by its
 * place and ringed in the chart series color at that place, the color the
 * section's cards take.
 */
export function trackOf(sections: readonly Section[] = SECTIONS): Stop[] {
  return [WORDS.start, ...sections.map((section, index) => ({
    href    : `/${section.name}/`,
    hue     : `var(${seriesProperty(index + 1)})`,
    label   : section.label,
    note    : section.question,
    numeral : numeralOf(index + 1)
  }))]
}
