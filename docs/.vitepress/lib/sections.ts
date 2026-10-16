/**
 * A section of the site placed in the registry, under the label the site names
 * it by and its place in the registry, counting from one.
 */
export interface Placed {
  label : string
  place : number
}

/**
 * A section of the site, meaning the folder under `docs/pages/` it opens on,
 * named `name`, the label the site names it by, and the question its pages
 * answer, which the page met at a missing address sets beside its label.
 */
export interface Section {
  label    : string
  name     : string
  question : string
}

/**
 * The sections of the site in the alphabetical order a registry keeps, each
 * one's place in the list numbering it from one.
 */
export const SECTIONS = [] as const satisfies readonly Section[]

/**
 * Writes a section's place in reading order as two digits.
 */
export const numeralOf = (place: number): string => String(place).padStart(2, '0')

/**
 * Finds the section the page at `relativePath` sits in, the one named by the
 * folder the page sits under, or `undefined` for a page under no folder the
 * registry names.
 *
 * Args:
 *   relativePath : The page's path beneath the pages' folder.
 *   sections     : The registry the folder is looked up in.
 */
export function sectionOf(relativePath: string, sections: readonly Section[] = SECTIONS): Placed | undefined {
  const folder = relativePath.split('/', 2)
  const index  = folder.length === 2 ? sections.findIndex(section => section.name === folder[0]) : -1
  return index === -1 ? undefined : { label: sections[index]!.label, place: index + 1 }
}
